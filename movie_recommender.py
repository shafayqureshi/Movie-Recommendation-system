import os
import threading
import time
import pickle
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# Default filenames (must be in same folder as this script)
# =========================
RATINGS_FILE_DEFAULT = "ratings.csv"
MOVIES_FILE_DEFAULT = "movies.csv"

# Cache files saved after training
MODEL_NPZ = "itemcf_model.npz"
MODEL_META = "itemcf_meta.pkl"

# =========================
# Mood -> Genre mapping
# =========================
MOOD_TO_GENRES = {
    "Action": ["Action", "Adventure"],
    "Comedy": ["Comedy"],
    "Romantic": ["Romance"],
    "Drama": ["Drama"],
    "Horror": ["Horror", "Thriller"],
    "Sci-Fi": ["Sci-Fi"],
    "Thriller": ["Thriller"],
    "Animation": ["Animation", "Family"],
    "Fantasy": ["Fantasy"],
    "Crime": ["Crime", "Mystery"],
    "Family": ["Family"],
    "Adventure": ["Adventure"],
}


# =========================
# Cache helpers
# =========================
def file_signature(path: str):
    if not os.path.exists(path):
        return None
    st = os.stat(path)
    return {"path": os.path.abspath(path), "mtime": st.st_mtime, "size": st.st_size}


def load_meta():
    if not os.path.exists(MODEL_META):
        return None
    try:
        with open(MODEL_META, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def save_meta(meta: dict):
    with open(MODEL_META, "wb") as f:
        pickle.dump(meta, f)


def cache_is_valid(ratings_path, movies_path, top_users, top_movies):
    meta = load_meta()
    if meta is None:
        return False

    if meta.get("top_users") != top_users or meta.get("top_movies") != top_movies:
        return False

    sig_r = file_signature(ratings_path)
    sig_m = file_signature(movies_path)
    if sig_r is None or sig_m is None:
        return False

    return (
        meta.get("ratings_sig") == sig_r
        and meta.get("movies_sig") == sig_m
        and os.path.exists(MODEL_NPZ)
    )


def save_model(ratings_path, movies_path, top_users, top_movies,
               item_ids, item_sim, item_mean, item_count, global_mean):
    np.savez_compressed(
        MODEL_NPZ,
        item_ids=item_ids,
        item_sim=item_sim,
        item_mean=item_mean,
        item_count=item_count,
        global_mean=np.array([global_mean], dtype=np.float32),
    )
    meta = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "top_users": top_users,
        "top_movies": top_movies,
        "ratings_sig": file_signature(ratings_path),
        "movies_sig": file_signature(movies_path),
    }
    save_meta(meta)


def load_model():
    data = np.load(MODEL_NPZ)
    item_ids = data["item_ids"].astype(np.int64)
    item_sim = data["item_sim"].astype(np.float32)
    item_mean = data["item_mean"].astype(np.float32)
    item_count = data["item_count"].astype(np.float32)
    global_mean = float(data["global_mean"][0])
    return item_ids, item_sim, item_mean, item_count, global_mean


# =========================
# Build the model (sampling + similarity + rating stats)
# =========================
def build_item_item_model(ratings_path, movies_path, top_users=3000, top_movies=1500, status_cb=None):
    """
    - Sample top active users and top popular movies (memory safe).
    - Compute item-item cosine similarity for similar-movie recommendations.
    - Compute per-item rating mean/count for mood/genre recommendations.
    """

    def say(msg):
        if status_cb:
            status_cb(msg)

    t0 = time.time()

    if not os.path.exists(ratings_path):
        raise FileNotFoundError(f"Cannot find {ratings_path}")
    if not os.path.exists(movies_path):
        raise FileNotFoundError(f"Cannot find {movies_path}")

    say("Loading ratings.csv ...")
    ratings = pd.read_csv(ratings_path, usecols=["userId", "movieId", "rating"])

    say("Loading movies.csv ...")
    movies = pd.read_csv(movies_path, usecols=["movieId", "title", "genres"])

    say(f"Sampling top {top_users} active users ...")
    top_user_ids = ratings["userId"].value_counts().head(top_users).index
    r1 = ratings[ratings["userId"].isin(top_user_ids)]

    say(f"Sampling top {top_movies} most-rated movies ...")
    top_movie_ids = r1["movieId"].value_counts().head(top_movies).index
    r2 = r1[r1["movieId"].isin(top_movie_ids)].copy()

    say(f"Sampled ratings: {len(r2):,} | users={r2['userId'].nunique():,} | movies={r2['movieId'].nunique():,}")

    say("Computing rating statistics ...")
    global_mean = float(r2["rating"].mean())
    stats = r2.groupby("movieId")["rating"].agg(["mean", "count"]).reset_index()
    stats.rename(columns={"mean": "item_mean", "count": "item_count"}, inplace=True)
    stats_map = stats.set_index("movieId")

    say("Building user-item matrix (pivot) ...")
    pivot = r2.pivot_table(index="userId", columns="movieId", values="rating", aggfunc="mean")
    item_ids = pivot.columns.to_numpy(dtype=np.int64)

    item_mean = np.array([float(stats_map.loc[mid, "item_mean"]) for mid in item_ids], dtype=np.float32)
    item_count = np.array([float(stats_map.loc[mid, "item_count"]) for mid in item_ids], dtype=np.float32)

    say("Preparing matrix (mean-centering, NaN -> 0) ...")
    X = pivot.to_numpy(dtype=np.float32)
    col_mean = np.nanmean(X, axis=0, keepdims=True).astype(np.float32)
    X = X - col_mean
    X = np.nan_to_num(X, nan=0.0)

    say("Computing item-item similarity (cosine) ...")
    item_sim = cosine_similarity(X.T).astype(np.float32)
    np.fill_diagonal(item_sim, 0.0)

    say(f"Done. Total time: {time.time() - t0:.1f}s")
    return item_ids, item_sim, item_mean, item_count, global_mean, movies


# =========================
# Recommendation functions
# =========================
def find_movie_matches(movies_df: pd.DataFrame, query: str, max_matches=40):
    q = (query or "").strip()
    if not q:
        return movies_df.iloc[0:0]
    mask = movies_df["title"].astype(str).str.contains(q, case=False, na=False)
    return movies_df[mask].head(max_matches).copy()


def recommend_similar_movies(item_ids, item_sim, movies_df, selected_movie_id: int, top_n=10):
    idx_map = {mid: i for i, mid in enumerate(item_ids)}
    if selected_movie_id not in idx_map:
        return None, "Selected movie is not in the sampled model. Try a more popular title."

    i = idx_map[selected_movie_id]
    sims = item_sim[i].copy()
    sims[i] = -1.0

    top_idx = np.argsort(-sims)[:top_n]
    rec_ids = item_ids[top_idx]
    rec_scores = sims[top_idx]

    id2title = dict(zip(movies_df["movieId"].astype(int), movies_df["title"].astype(str)))
    results = [(int(mid), id2title.get(int(mid), str(int(mid))), float(score)) for mid, score in zip(rec_ids, rec_scores)]
    return results, None


def recommend_by_mood(item_ids, item_mean, item_count, global_mean, movies_df,
                      mood: str, top_n=15, min_votes=50):
    genres = MOOD_TO_GENRES.get(mood, [])
    if not genres:
        return None, "Invalid mood."

    stats_map = {int(mid): (float(r), float(v)) for mid, r, v in zip(item_ids, item_mean, item_count)}

    results = []
    for row in movies_df.itertuples(index=False):
        mid = int(row.movieId)
        if mid not in stats_map:
            continue
        g = str(row.genres)
        if any(gg in g for gg in genres):
            R, v = stats_map[mid]
            m = float(min_votes)
            score = (v / (v + m)) * R + (m / (v + m)) * global_mean
            results.append((mid, str(row.title), score, R, v, g))

    results.sort(key=lambda x: x[2], reverse=True)
    return results[:top_n], None if results else ("", "No results. Increase Top movies or change mood.")


# =========================
# GUI App (Dark theme)
# =========================
class MovieRecommenderApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("IR & DM Final Project — Movie Recommender (Mood + Similar Movies)")
        self.geometry("1050x700")
        self.minsize(980, 640)

        # Dark palette
        self.bg = "#0f111a"
        self.panel = "#141824"
        self.card = "#171c2a"
        self.text = "#e6e6e6"
        self.muted = "#b7bcc8"
        self.accent = "#7aa2f7"

        self.configure(bg=self.bg)

        # Model state
        self.item_ids = None
        self.item_sim = None
        self.item_mean = None
        self.item_count = None
        self.global_mean = None
        self.movies_df = None
        self.selected_movie_id = None

        self._build_styles()
        self._build_layout()
        self._update_status("Ready. Click 'Load/Train Model'. First run trains and saves; next run loads fast.")

    def _build_styles(self):
        style = ttk.Style(self)
        style.theme_use("clam")

        style.configure("TFrame", background=self.bg)
        style.configure("Panel.TFrame", background=self.panel)
        style.configure("Card.TFrame", background=self.card)

        style.configure("TLabel", background=self.bg, foreground=self.text, font=("Segoe UI", 10))
        style.configure("Title.TLabel", background=self.bg, foreground=self.text, font=("Segoe UI", 16, "bold"))
        style.configure("Muted.TLabel", background=self.bg, foreground=self.muted, font=("Segoe UI", 10))
        style.configure("Section.TLabel", background=self.card, foreground=self.text, font=("Segoe UI", 12, "bold"))

        style.configure("TEntry", fieldbackground=self.card, background=self.card, foreground=self.text)
        style.configure("TCombobox", fieldbackground=self.card, background=self.card, foreground=self.text)
        style.configure("TButton", background=self.accent, foreground="#0b0d14", font=("Segoe UI", 10, "bold"))
        style.map("TButton", background=[("active", "#9bb4f7")])
        style.configure("TProgressbar", background=self.accent, troughcolor=self.card)

    def _build_layout(self):
        # Header
        header = ttk.Frame(self, style="TFrame")
        header.pack(fill="x", padx=18, pady=(16, 8))

        ttk.Label(header, text="Movie Recommendation System", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            header,
            text="Modes: (1) Mood/Genre recommendations (2) Similar movies using item-to-item CF. Model is cached to disk.",
            style="Muted.TLabel"
        ).pack(anchor="w", pady=(2, 0))

        main = ttk.Frame(self, style="TFrame")
        main.pack(fill="both", expand=True, padx=18, pady=12)

        left = ttk.Frame(main, style="Panel.TFrame")
        left.pack(side="left", fill="y", padx=(0, 10))

        right = ttk.Frame(main, style="Panel.TFrame")
        right.pack(side="right", fill="both", expand=True, padx=(10, 0))

        # ---- Left: model setup
        controls = ttk.Frame(left, style="Panel.TFrame")
        controls.pack(fill="x", padx=14, pady=14)

        ttk.Label(controls, text="Model Setup", style="Title.TLabel").pack(anchor="w", pady=(0, 8))

        ttk.Label(controls, text="ratings file:", style="Muted.TLabel").pack(anchor="w")
        self.ratings_var = tk.StringVar(value=RATINGS_FILE_DEFAULT)
        ttk.Entry(controls, textvariable=self.ratings_var).pack(fill="x", pady=(2, 10))

        ttk.Label(controls, text="movies file:", style="Muted.TLabel").pack(anchor="w")
        self.movies_var = tk.StringVar(value=MOVIES_FILE_DEFAULT)
        ttk.Entry(controls, textvariable=self.movies_var).pack(fill="x", pady=(2, 12))

        ttk.Label(controls, text="Sampling (recommended for MovieLens-32M):", style="Muted.TLabel").pack(anchor="w")

        row = ttk.Frame(controls, style="Panel.TFrame")
        row.pack(fill="x", pady=(6, 10))

        ttk.Label(row, text="Top users:", style="Muted.TLabel").grid(row=0, column=0, sticky="w")
        self.top_users_var = tk.IntVar(value=3000)
        ttk.Entry(row, textvariable=self.top_users_var, width=10).grid(row=0, column=1, padx=(8, 16))

        ttk.Label(row, text="Top movies:", style="Muted.TLabel").grid(row=0, column=2, sticky="w")
        self.top_movies_var = tk.IntVar(value=1500)
        ttk.Entry(row, textvariable=self.top_movies_var, width=10).grid(row=0, column=3, padx=(8, 0))

        btns = ttk.Frame(controls, style="Panel.TFrame")
        btns.pack(fill="x", pady=(6, 0))

        self.train_btn = ttk.Button(btns, text="Load/Train Model", command=self.on_train_clicked)
        self.train_btn.pack(fill="x", pady=(0, 8))

        self.force_btn = ttk.Button(btns, text="Force Retrain", command=lambda: self.on_train_clicked(force=True))
        self.force_btn.pack(fill="x")

        self.progress = ttk.Progressbar(left, mode="indeterminate")
        self.progress.pack(fill="x", padx=14, pady=(0, 8))

        self.status_var = tk.StringVar(value="")
        ttk.Label(left, textvariable=self.status_var, style="Muted.TLabel", wraplength=280).pack(fill="x", padx=14, pady=(0, 14))

        # ---- Right: tabs
        nb = ttk.Notebook(right)
        nb.pack(fill="both", expand=True, padx=14, pady=14)

        # Mood tab
        tab_mood = ttk.Frame(nb, style="Panel.TFrame")
        nb.add(tab_mood, text="Mood/Genre")

        mood_card = ttk.Frame(tab_mood, style="Card.TFrame")
        mood_card.pack(fill="x", pady=(0, 12))

        ttk.Label(mood_card, text="Mood-based Recommendation", style="Section.TLabel").pack(anchor="w", padx=12, pady=(10, 6))
        ttk.Label(mood_card, text="Select a mood; system filters by genre and ranks using ratings-based Bayesian score.", style="Muted.TLabel").pack(anchor="w", padx=12, pady=(0, 10))

        mood_row = ttk.Frame(mood_card, style="Card.TFrame")
        mood_row.pack(fill="x", padx=12, pady=(0, 12))

        self.mood_var = tk.StringVar(value="Action")
        self.mood_box = ttk.Combobox(mood_row, textvariable=self.mood_var, values=list(MOOD_TO_GENRES.keys()), state="readonly")
        self.mood_box.pack(side="left")

        ttk.Label(mood_row, text="Min votes:", style="Muted.TLabel").pack(side="left", padx=(16, 6))
        self.min_votes_var = tk.IntVar(value=50)
        ttk.Entry(mood_row, textvariable=self.min_votes_var, width=8).pack(side="left")

        self.mood_btn = ttk.Button(mood_row, text="Recommend", command=self.on_mood_recommend)
        self.mood_btn.pack(side="left", padx=(16, 0))

        self.mood_out = tk.Text(tab_mood, bg=self.card, fg=self.text, insertbackground=self.text,
                                relief="flat", highlightthickness=0, font=("Consolas", 10))
        self.mood_out.pack(fill="both", expand=True)
        self._set_text(self.mood_out, "Load/train the model first.\n")

        # Similar tab
        tab_sim = ttk.Frame(nb, style="Panel.TFrame")
        nb.add(tab_sim, text="Because you liked...")

        sim_card = ttk.Frame(tab_sim, style="Card.TFrame")
        sim_card.pack(fill="x", pady=(0, 12))

        ttk.Label(sim_card, text="Similar-Movie Recommendation", style="Section.TLabel").pack(anchor="w", padx=12, pady=(10, 6))
        ttk.Label(sim_card, text="Type a movie keyword, select a match, then recommend similar movies.", style="Muted.TLabel").pack(anchor="w", padx=12, pady=(0, 10))

        sim_row = ttk.Frame(sim_card, style="Card.TFrame")
        sim_row.pack(fill="x", padx=12, pady=(0, 12))

        self.query_var = tk.StringVar(value="")
        self.query_entry = ttk.Entry(sim_row, textvariable=self.query_var)
        self.query_entry.pack(side="left", fill="x", expand=True)
        self.query_entry.bind("<KeyRelease>", lambda e: self.refresh_matches())

        self.sim_btn = ttk.Button(sim_row, text="Recommend", command=self.on_sim_recommend)
        self.sim_btn.pack(side="left", padx=(10, 0))

        bottom = ttk.Frame(tab_sim, style="Panel.TFrame")
        bottom.pack(fill="both", expand=True)

        matches_frame = ttk.Frame(bottom, style="Card.TFrame")
        matches_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))

        ttk.Label(matches_frame, text="Matches", style="Section.TLabel").pack(anchor="w", padx=12, pady=(10, 6))
        self.matches_list = tk.Listbox(matches_frame, bg=self.card, fg=self.text,
                                       selectbackground=self.accent, selectforeground="#0b0d14",
                                       highlightthickness=0, relief="flat", activestyle="none",
                                       font=("Segoe UI", 10))
        self.matches_list.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        self.matches_list.bind("<<ListboxSelect>>", self.on_match_selected)

        rec_frame = ttk.Frame(bottom, style="Card.TFrame")
        rec_frame.pack(side="right", fill="both", expand=True, padx=(10, 0))

        ttk.Label(rec_frame, text="Recommendations", style="Section.TLabel").pack(anchor="w", padx=12, pady=(10, 6))
        self.sim_out = tk.Text(rec_frame, bg=self.card, fg=self.text, insertbackground=self.text,
                               relief="flat", highlightthickness=0, font=("Consolas", 10))
        self.sim_out.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        self._set_text(self.sim_out, "Load/train the model first.\n")

    def _set_text(self, widget, text):
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("end", text)
        widget.configure(state="disabled")

    def _update_status(self, msg):
        self.status_var.set(msg)

    # ---------------- Training/load
    def on_train_clicked(self, force=False):
        ratings_path = self.ratings_var.get().strip()
        movies_path = self.movies_var.get().strip()
        top_users = int(self.top_users_var.get())
        top_movies = int(self.top_movies_var.get())

        # Ensure we run relative to script directory (important!)
        # This makes it find ratings.csv even if you run python from another folder.
        script_dir = os.path.dirname(os.path.abspath(__file__))
        ratings_path = os.path.join(script_dir, ratings_path)
        movies_path = os.path.join(script_dir, movies_path)

        if not force and cache_is_valid(ratings_path, movies_path, top_users, top_movies):
            try:
                self.progress.start(12)
                self._update_status("Loading cached model ...")
                self.item_ids, self.item_sim, self.item_mean, self.item_count, self.global_mean = load_model()
                self.movies_df = pd.read_csv(movies_path, usecols=["movieId", "title", "genres"])
                self.progress.stop()
                self._update_status("Cached model loaded. Ready.")
                return
            except Exception as ex:
                self.progress.stop()
                messagebox.showwarning("Cache load failed", f"Cache could not be loaded.\nRetraining.\n\n{ex}")

        self.train_btn.configure(state="disabled")
        self.force_btn.configure(state="disabled")
        self.progress.start(12)
        self._update_status("Training... (first time may take a few minutes)")

        def worker():
            try:
                item_ids, item_sim, item_mean, item_count, global_mean, movies_df = build_item_item_model(
                    ratings_path, movies_path,
                    top_users=top_users, top_movies=top_movies,
                    status_cb=lambda m: self.after(0, self._update_status, m)
                )

                save_model(ratings_path, movies_path, top_users, top_movies,
                           item_ids, item_sim, item_mean, item_count, global_mean)

                def finish():
                    self.item_ids = item_ids
                    self.item_sim = item_sim
                    self.item_mean = item_mean
                    self.item_count = item_count
                    self.global_mean = global_mean
                    self.movies_df = movies_df
                    self.progress.stop()
                    self.train_btn.configure(state="normal")
                    self.force_btn.configure(state="normal")
                    self._update_status("Training complete. Model saved. Ready for recommendations.")

                self.after(0, finish)

            except Exception as ex:
                err_msg = str(ex)

                def fail(msg=err_msg):
                    self.progress.stop()
                    self.train_btn.configure(state="normal")
                    self.force_btn.configure(state="normal")
                    self._update_status("Training failed.")
                    messagebox.showerror("Training failed", msg)

                print("TRAINING ERROR:", err_msg)
                self.after(0, fail)

        threading.Thread(target=worker, daemon=True).start()

    # ---------------- Similar tab handlers
    def refresh_matches(self):
        if self.movies_df is None:
            return
        q = self.query_var.get().strip()
        matches = find_movie_matches(self.movies_df, q, max_matches=40)

        self.matches_list.delete(0, "end")
        self.selected_movie_id = None

        for _, row in matches.iterrows():
            mid = int(row["movieId"])
            title = str(row["title"])
            self.matches_list.insert("end", f"{title}  [movieId={mid}]")

    def on_match_selected(self, _event=None):
        sel = self.matches_list.curselection()
        if not sel:
            self.selected_movie_id = None
            return
        text = self.matches_list.get(sel[0])
        try:
            mid_str = text.split("movieId=")[-1].rstrip("]")
            self.selected_movie_id = int(mid_str)
        except Exception:
            self.selected_movie_id = None

    def on_sim_recommend(self):
        if self.item_ids is None:
            messagebox.showinfo("Model not ready", "Load/Train the model first.")
            return

        if self.selected_movie_id is None:
            if self.matches_list.size() > 0:
                self.matches_list.selection_set(0)
                self.matches_list.activate(0)
                self.on_match_selected()
            else:
                messagebox.showinfo("No match", "No matching movie found. Try another keyword.")
                return

        results, err = recommend_similar_movies(self.item_ids, self.item_sim, self.movies_df,
                                                self.selected_movie_id, top_n=10)
        if err:
            self._set_text(self.sim_out, err + "\nTip: try Star Wars, Toy Story, Batman, Matrix.\n")
            return

        sel_title = self.movies_df.loc[self.movies_df["movieId"].astype(int) == int(self.selected_movie_id), "title"]
        sel_title = sel_title.iloc[0] if len(sel_title) else str(self.selected_movie_id)

        lines = [f"Because you selected: {sel_title} (movieId={self.selected_movie_id})\n",
                 "Top similar movies:\n"]
        for i, (mid, title, score) in enumerate(results, start=1):
            lines.append(f"{i:>2}. {title:<60}  similarity={score:.3f}  movieId={mid}")

        self._set_text(self.sim_out, "\n".join(lines))

    # ---------------- Mood tab handler
    def on_mood_recommend(self):
        if self.item_ids is None:
            messagebox.showinfo("Model not ready", "Load/Train the model first.")
            return

        mood = self.mood_var.get().strip()
        min_votes = int(self.min_votes_var.get())

        results, err = recommend_by_mood(
            self.item_ids, self.item_mean, self.item_count, self.global_mean,
            self.movies_df, mood=mood, top_n=15, min_votes=min_votes
        )

        if results is None or len(results) == 0:
            self._set_text(self.mood_out, err if err else "No results.")
            return

        lines = [
            f"Mood: {mood}",
            f"Genres: {', '.join(MOOD_TO_GENRES[mood])}",
            f"Ranking: Bayesian average (min_votes={min_votes})\n",
            "Recommended movies:\n",
        ]

        for i, (mid, title, bayes, mean_r, cnt, genres) in enumerate(results, start=1):
            lines.append(
                f"{i:>2}. {title:<55}  score={bayes:.3f}  mean={mean_r:.2f}  votes={int(cnt):>6}  movieId={mid}"
            )

        self._set_text(self.mood_out, "\n".join(lines))


def main():
    app = MovieRecommenderApp()
    app.mainloop()


if __name__ == "__main__":
    main()
