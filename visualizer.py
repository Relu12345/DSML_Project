import math
import os
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import chess
import chess.pgn
import chess.engine
import pandas as pd
import numpy as np
import seaborn as sns

###############################################################################
#                            CHESS / STOCKFISH LOGIC                          #
###############################################################################

def load_pgn_games(pgn_path, max_games=2):
    """
    Load up to 'max_games' from the PGN file and return a list of chess.pgn.Game objects.
    """
    games = []
    with open(pgn_path, "r", encoding="utf-8") as pgn:
        for _ in range(max_games):
            game = chess.pgn.read_game(pgn)
            if game is None:
                break
            games.append(game)
    return games

def evaluate_game_with_stockfish(game, stockfish_path, depth=15):
    """
    Run Stockfish evaluation for each move of the given game.
    Returns a list of dicts, each with 'move_uci', 'eval_white', 'eval_black'.
    """
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    board = game.board()

    evaluations = []
    for move in game.mainline_moves():
        board.push(move)
        info = engine.analyse(board, limit=chess.engine.Limit(depth=depth))
        eval_white = info["score"].white().score(mate_score=10000)
        eval_black = info["score"].black().score(mate_score=10000)

        evaluations.append({
            "move_uci": move.uci(),
            "eval_white": eval_white,
            "eval_black": eval_black,
        })

    engine.quit()
    return evaluations

def calculate_entropy_for_game(game):
    """
    Compute log(#legal_moves + 1) for each position in the game.
    Returns a list of entropies corresponding to each move in the game.
    """
    board = game.board()
    entropies = []

    for move in game.mainline_moves():
        legal_moves_count = len(list(board.legal_moves))
        ent = math.log(legal_moves_count + 1)  # Entropy = ln(#legal_moves + 1)
        entropies.append(ent)
        board.push(move)

    return entropies

def get_stockfish_best_move(board, stockfish_path, depth=15):
    """
    Get the best move (as a Move object) for the current board from Stockfish.
    """
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    result = engine.play(board, limit=chess.engine.Limit(depth=depth))
    engine.quit()
    return result.move

###############################################################################
#                  HELPER FUNCTIONS FOR BOARD RENDERING                       #
###############################################################################

SQUARE_SIZE = 32
MARGIN = 10
BOARD_SIZE = 8 * SQUARE_SIZE + 2 * MARGIN  # e.g., 8*32 + 2*10 = 296

def render_board_image(board):
    from PIL import ImageDraw
    img = Image.new("RGBA", (BOARD_SIZE, BOARD_SIZE), color=(255, 255, 255, 0))
    draw = ImageDraw.Draw(img)

    # Draw squares
    for row in range(8):
        for col in range(8):
            color = (240, 217, 181) if (row + col) % 2 == 0 else (181, 136, 99)
            x1 = MARGIN + col * SQUARE_SIZE
            y1 = MARGIN + row * SQUARE_SIZE
            x2 = x1 + SQUARE_SIZE
            y2 = y1 + SQUARE_SIZE
            draw.rectangle([x1, y1, x2, y2], fill=color)

    # Draw pieces
    for row in range(8):
        for col in range(8):
            piece = board.piece_at(chess.square(col, 7 - row))
            if piece:
                piece_image = load_piece_image(piece.symbol(), SQUARE_SIZE)
                if piece_image:
                    x = MARGIN + col * SQUARE_SIZE
                    y = MARGIN + row * SQUARE_SIZE
                    img.paste(piece_image, (x, y), piece_image)

    # Add file/rank labels
    for c in range(8):
        letter = chr(ord('a') + c)
        draw.text((MARGIN + c*SQUARE_SIZE + 12, BOARD_SIZE - MARGIN + 2), letter, fill=(0,0,0))
        draw.text((MARGIN + c*SQUARE_SIZE + 12, 2), letter, fill=(0,0,0))

    for r in range(8):
        number = str(8 - r)
        draw.text((2, MARGIN + r*SQUARE_SIZE + 10), number, fill=(0,0,0))
        draw.text((BOARD_SIZE - MARGIN + 2, MARGIN + r*SQUARE_SIZE + 10), number, fill=(0,0,0))

    return img

def load_piece_image(piece_symbol, square_size):
    base_dir = "images"  # Adjust as needed
    piece_map = {
        'p': 'pawn', 'n': 'knight', 'b': 'bishop',
        'r': 'rook', 'q': 'queen', 'k': 'king'
    }
    color = 'white' if piece_symbol.isupper() else 'black'
    piece_type = piece_map[piece_symbol.lower()]
    filename = f"{color}_{piece_type}.png"
    path = os.path.join(base_dir, filename)

    if not os.path.exists(path):
        return None

    try:
        img = Image.open(path).convert("RGBA")
        # Pillow 9.1+ => Image.Resampling.LANCZOS; 
        # for older Pillow versions, use Image.LANCZOS
        img = img.resize((square_size, square_size), Image.Resampling.LANCZOS)
        return img
    except Exception as e:
        print(f"Could not load piece image {path}: {e}")
        return None

###############################################################################
#                   SCROLLABLE FRAME IMPLEMENTATION                           #
###############################################################################
class ScrollableFrame(tk.Frame):
    """
    A reusable scrollable container (Canvas + vertical scrollbar).
    Everything you place inside 'self.scrollable_frame' becomes scrollable.
    """
    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)

        self.canvas = tk.Canvas(self, bg="#fafafa")
        self.scrollbar = tk.Scrollbar(self, orient="vertical", command=self.canvas.yview)

        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # The actual frame that holds widgets
        self.scrollable_frame = tk.Frame(self.canvas)
        self.scrollable_frame_id = self.canvas.create_window(
            (0, 0), window=self.scrollable_frame, anchor="nw"
        )

        self.scrollable_frame.bind("<Configure>", self._on_frame_configure)
        # Optional: bind mouse wheel to scroll
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

    def _on_frame_configure(self, event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_mousewheel(self, event):
        """
        Windows typically uses event.delta = 120 or -120.
        On Linux or Mac, you may need different handling.
        """
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

###############################################################################
#                     THE MAIN VISUALIZER / TKINTER APP                       #
###############################################################################

class ChessMoveEvaluator:
    """
    - Shows 3 rows of boards: Real, Stockfish, User.
    - We also provide additional plots:
      * Evaluation Over Time
      * Entropy Trends
      * Deviation Maps (heatmap of D_Actual-AI, D_Actual-Friend, D_AI-Friend)
      * Chaos Index (std dev of [G_Actual, G_AI, G_Friend])
      * Best vs. Played Outcome (Delta_AI, Delta_Friend)
      * Chess Chaos Art (scatter plot of ChaosIndex vs move #, colored by D_Actual-AI)
      * Correlation Analysis (heatmap of correlation among [G_Actual, G_AI, G_Friend, entropy, ChaosIndex])
    """

    def __init__(self, root, games, stockfish_path):
        self.root = root
        self.root.title("Chess Move Evaluator")
        self.root.geometry("1200x900")

        self.games = games
        self.stockfish_path = stockfish_path
        self.current_game_index = 0

        # We'll store (game, df_evals) in game_data
        self.game_data = []
        for gm in self.games:
            df = self._evaluate_game_and_build_df(gm)
            self.game_data.append((gm, df))

        # Keep track of user moves' entropies (for plotting as red dots).
        self.user_moves_entropy_data = []

        #######################################################################
        # 1) TOP FRAME: Title
        #######################################################################
        top_title_frame = tk.Frame(self.root)
        top_title_frame.pack(side=tk.TOP, fill=tk.X, pady=5)

        self.game_title_label = tk.Label(top_title_frame, text="Game Title", font=("Arial", 16, "bold"))
        self.game_title_label.pack(side=tk.TOP, pady=(0,5))

        #######################################################################
        # 2) SUB-FRAME FOR GAME & MOVE BUTTONS (UNDER TITLE)
        #######################################################################
        nav_game_move_frame = tk.Frame(top_title_frame)
        nav_game_move_frame.pack(side=tk.TOP, fill=tk.X)

        # Prev Game / Next Game
        self.prev_game_btn = tk.Button(nav_game_move_frame, text="<< Prev Game", command=self.prev_game)
        self.prev_game_btn.pack(side=tk.LEFT, padx=5)

        self.next_game_btn = tk.Button(nav_game_move_frame, text="Next Game >>", command=self.next_game)
        self.next_game_btn.pack(side=tk.LEFT, padx=5)

        # Prev Move / Next Move
        self.prev_move_btn = tk.Button(nav_game_move_frame, text="<< Prev Move", command=self.prev_move)
        self.prev_move_btn.pack(side=tk.LEFT, padx=5)

        self.next_move_btn = tk.Button(nav_game_move_frame, text="Next Move >>", command=self.next_move)
        self.next_move_btn.pack(side=tk.LEFT, padx=5)

        # Move # field
        self.move_num_var = tk.IntVar(value=1)
        tk.Label(nav_game_move_frame, text="Move #:").pack(side=tk.LEFT, padx=(10,0))
        self.move_num_entry = tk.Entry(nav_game_move_frame, width=4, textvariable=self.move_num_var)
        self.move_num_entry.pack(side=tk.LEFT, padx=5)

        self.go_button = tk.Button(nav_game_move_frame, text="Go", command=self.go_to_move)
        self.go_button.pack(side=tk.LEFT, padx=5)

        #######################################################################
        # 3) LEFT: SCROLLABLE (3 rows)
        #######################################################################
        self.scroll_frame = ScrollableFrame(self.root)
        self.scroll_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.real_move_row = self.add_move_section(
            parent=self.scroll_frame.scrollable_frame,
            label_text="Real Game Move",
            show_user_input=False
        )
        self.sf_move_row = self.add_move_section(
            parent=self.scroll_frame.scrollable_frame,
            label_text="Stockfish Best Move",
            show_user_input=False
        )
        self.user_move_row = self.add_move_section(
            parent=self.scroll_frame.scrollable_frame,
            label_text="Your Move",
            show_user_input=True
        )

        #######################################################################
        # 4) RIGHT: Plot area
        #######################################################################
        right_frame = tk.Frame(self.root, padx=10, pady=10)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(0, 10))

        tk.Label(right_frame, text="Select Plot:", font=("Arial", 14)).grid(row=0, column=0, sticky="w")

        # Add new plot options, including Deviation Maps, Chaos Index, etc.
        self.plot_dropdown = ttk.Combobox(
            right_frame, 
            values=[
                "Evaluation Over Time", 
                "Entropy Trends", 
                "Deviation Maps",
                "Chaos Index",
                "Best vs. Played Outcome",
                "Chess Chaos Art",
                "Correlation Analysis"
            ], 
            state="readonly"
        )
        self.plot_dropdown.grid(row=0, column=1, sticky="w")
        self.plot_dropdown.set("Select Plot")
        self.plot_dropdown.bind("<<ComboboxSelected>>", self.update_plot)

        self.figure = plt.Figure(figsize=(7, 6), dpi=75)
        self.mpl_canvas = FigureCanvasTkAgg(self.figure, master=right_frame)
        self.mpl_canvas.get_tk_widget().grid(row=1, column=0, columnspan=2, pady=10, sticky="nsew")
        right_frame.grid_columnconfigure(0, weight=1)
        right_frame.grid_columnconfigure(1, weight=1)
        right_frame.grid_rowconfigure(1, weight=1)

        # Load the first game
        self.load_game(0)

    def _evaluate_game_and_build_df(self, game):
        evals = evaluate_game_with_stockfish(game, self.stockfish_path, depth=15)
        entropies = calculate_entropy_for_game(game)
        df = pd.DataFrame(evals)
        df["entropy"] = entropies

        # G_Actual: from white's perspective => eval_white
        df["G_Actual"] = df["eval_white"]
        # G_AI: from white's perspective => -eval_black
        df["G_AI"] = -df["eval_black"]
        
        # Initialize G_Friend as NaN (no user move data yet)
        df["G_Friend"] = np.nan

        # Deviation Maps (these will be updated once we have G_Friend for each row)
        df["D_Actual_AI"] = np.abs(df["G_Actual"] - df["G_AI"])
        df["D_Actual_Friend"] = np.nan
        df["D_AI_Friend"] = np.nan

        # Chaos Index
        df["ChaosIndex"] = np.nan

        # Best vs. Played Outcome
        df["Delta_AI"] = df["G_AI"] - df["G_Actual"]
        df["Delta_Friend"] = np.nan

        return df


    ###########################################################################
    #                            GAME LOADING                                  #
    ###########################################################################
    def load_game(self, idx):
        if idx < 0 or idx >= len(self.games):
            return

        self.current_game_index = idx
        self.game, self.evals_df = self.game_data[idx]
        self.all_moves = list(self.game.mainline_moves())
        self.current_move_index = 0

        # Reset user moves so old user entropies don't carry over to a new game
        self.user_moves_entropy_data = []

        self.show_game_title()
        self.initialize_chess_boards()
        self.update_all_boards()

    def prev_game(self):
        if self.current_game_index > 0:
            self.load_game(self.current_game_index - 1)

    def next_game(self):
        if self.current_game_index < len(self.games) - 1:
            self.load_game(self.current_game_index + 1)

    def show_game_title(self):
        headers = self.game.headers
        white = headers.get("White", "Unknown")
        black = headers.get("Black", "Unknown")
        date = headers.get("UTCDate", "")
        time = headers.get("UTCTime", "")
        title_text = f"{white} vs {black} - {date} {time}"
        self.game_title_label.config(text=title_text)

    ###########################################################################
    #                          MOVE NAVIGATION                                #
    ###########################################################################
    def prev_move(self):
        if self.current_move_index > 0:
            self.current_move_index -= 1
            self.update_all_boards()
            self.move_num_var.set(self.current_move_index + 1)

    def next_move(self):
        if self.current_move_index < len(self.all_moves):
            self.current_move_index += 1
            # clamp if we exceed
            if self.current_move_index > len(self.all_moves):
                self.current_move_index = len(self.all_moves)
            self.update_all_boards()
            self.move_num_var.set(self.current_move_index + 1)

    def go_to_move(self):
        requested_move = self.move_num_var.get()
        if requested_move < 1:
            requested_move = 1
        if requested_move > len(self.all_moves):
            requested_move = len(self.all_moves)

        self.current_move_index = requested_move - 1
        self.update_all_boards()
        self.move_num_var.set(requested_move)

    ###########################################################################
    #                             ROW CREATION                                #
    ###########################################################################
    def add_move_section(self, parent, label_text, show_user_input=False):
        section_frame = tk.Frame(parent, pady=10)
        section_frame.pack(fill=tk.X, expand=True)

        tk.Label(section_frame, text=label_text, font=("Arial", 16)).pack(anchor="w")

        boards_frame = tk.Frame(section_frame)
        boards_frame.pack(fill=tk.X, expand=True)

        before_canvas = tk.Canvas(boards_frame, width=BOARD_SIZE, height=BOARD_SIZE, bg="white")
        before_canvas.pack(side=tk.LEFT, padx=5)

        after_canvas = tk.Canvas(boards_frame, width=BOARD_SIZE, height=BOARD_SIZE, bg="white")
        after_canvas.pack(side=tk.LEFT, padx=5)

        ge_frame = tk.Frame(boards_frame, padx=10)
        ge_frame.pack(side=tk.LEFT, fill=tk.Y)

        tk.Label(ge_frame, text="Grade:", font=("Arial", 12)).pack(anchor="w")
        grade_label = tk.Label(ge_frame, text="0.00", font=("Arial", 12))
        grade_label.pack(anchor="w", pady=(0, 10))

        tk.Label(ge_frame, text="Entropy:", font=("Arial", 12)).pack(anchor="w")
        entropy_label = tk.Label(ge_frame, text="0.00", font=("Arial", 12))
        entropy_label.pack(anchor="w")

        notation_entry = None
        calc_button = None
        if show_user_input:
            user_input_frame = tk.Frame(section_frame, pady=5)
            user_input_frame.pack(fill=tk.X, expand=True)

            tk.Label(user_input_frame, text="Enter move (e.g. e4, Nf3): ").pack(side=tk.LEFT)
            notation_entry = tk.Entry(user_input_frame, width=10)
            notation_entry.pack(side=tk.LEFT, padx=5)

            calc_button = tk.Button(
                user_input_frame, text="Calculate",
                command=lambda: self.on_user_move_calculate(
                    notation_entry, before_canvas, after_canvas, grade_label, entropy_label
                )
            )
            calc_button.pack(side=tk.LEFT)

        return {
            "frame": section_frame,
            "before_canvas": before_canvas,
            "after_canvas": after_canvas,
            "grade_label": grade_label,
            "entropy_label": entropy_label,
            "notation_entry": notation_entry,
            "calc_button": calc_button
        }

    def initialize_chess_boards(self):
        for row in (self.real_move_row, self.sf_move_row, self.user_move_row):
            row["before_canvas"].delete("all")
            row["before_canvas"].create_text(
                BOARD_SIZE//2, BOARD_SIZE//2,
                text="Before\nBoard", font=("Arial", 14)
            )
            row["after_canvas"].delete("all")
            row["after_canvas"].create_text(
                BOARD_SIZE//2, BOARD_SIZE//2,
                text="After\nBoard", font=("Arial", 14)
            )

    def draw_board_on_canvas(self, canvas, board):
        canvas.delete("all")
        img_pil = render_board_image(board)
        img_tk = ImageTk.PhotoImage(img_pil)
        canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
        canvas.image = img_tk

    ###########################################################################
    #                      UPDATE ALL 3 BOARDS FOR CURRENT MOVE               #
    ###########################################################################
    def update_all_boards(self):
        # 1) Identify the position "before" the real move
        real_before_board = self.game.board()
        for m in self.all_moves[: self.current_move_index]:
            real_before_board.push(m)

        # Build the "after" board for the real move
        real_after_board = real_before_board.copy()
        real_move = None
        if 0 <= self.current_move_index < len(self.all_moves):
            real_move = self.all_moves[self.current_move_index]
            real_after_board.push(real_move)

        # 2) Real Boards
        self.draw_board_on_canvas(self.real_move_row["before_canvas"], real_before_board)
        self.draw_board_on_canvas(self.real_move_row["after_canvas"], real_after_board)

        # Evaluate Real Move Grade
        real_eval, best_eval = self._compare_real_vs_best(real_before_board, real_move)
        real_grade = real_eval - best_eval
        # Entropy from the AFTER position of the real move
        real_entropy = self._calculate_entropy(real_after_board)

        self.real_move_row["grade_label"].config(text=f"{real_grade:.2f}")
        self.real_move_row["entropy_label"].config(text=f"{real_entropy:.2f}")

        # 3) Stockfish Boards
        sf_before_board = real_before_board.copy()
        sf_best_move = get_stockfish_best_move(sf_before_board, self.stockfish_path) if sf_before_board is not None else None
        sf_after_board = sf_before_board.copy()
        if sf_best_move:
            sf_after_board.push(sf_best_move)

        self.draw_board_on_canvas(self.sf_move_row["before_canvas"], sf_before_board)
        self.draw_board_on_canvas(self.sf_move_row["after_canvas"], sf_after_board)

        # Evaluate Stockfish Move Grade
        sf_eval, best_eval2 = self._compare_real_vs_best(sf_before_board, sf_best_move)

        # We'll define stockfish grade = sf_eval - real_eval_sfpos 
        real_eval_sfpos = 0.0
        if real_move and real_move in sf_before_board.legal_moves:
            tmp2 = sf_before_board.copy()
            tmp2.push(real_move)
            real_eval_sfpos = self._evaluate_position(tmp2)

        sf_grade = sf_eval - real_eval_sfpos
        # Entropy from the AFTER position of stockfish's move
        sf_entropy = self._calculate_entropy(sf_after_board)

        self.sf_move_row["grade_label"].config(text=f"{sf_grade:.2f}")
        self.sf_move_row["entropy_label"].config(text=f"{sf_entropy:.2f}")

        # 4) User boards
        #   - We do not push anything yet; user will push in on_user_move_calculate
        self.user_board = real_before_board.copy()  # "before" the user move
        user_after_board = self.user_board.copy()

        self.draw_board_on_canvas(self.user_move_row["before_canvas"], self.user_board)
        self.draw_board_on_canvas(self.user_move_row["after_canvas"], user_after_board)
        # For now, grade = 0, entropy = 0 (until user actually makes a move)
        self.user_move_row["grade_label"].config(text="0.00")
        self.user_move_row["entropy_label"].config(text="0.00")

        # If the user has selected a plot, refresh it
        if self.plot_dropdown.get() != "Select Plot":
            self.update_plot(None)

    def _compare_real_vs_best(self, board_before, move):
        """
        Evaluate 'move' from board_before => move_eval,
        Evaluate best move from same board => best_eval,
        return (move_eval, best_eval).
        """
        if move and move in board_before.legal_moves:
            temp = board_before.copy()
            temp.push(move)
            move_eval = self._evaluate_position(temp)
        else:
            move_eval = 0.0

        best_m = get_stockfish_best_move(board_before, self.stockfish_path)
        best_eval = 0.0
        if best_m:
            temp2 = board_before.copy()
            temp2.push(best_m)
            best_eval = self._evaluate_position(temp2)

        return move_eval, best_eval

    def _evaluate_position(self, board, depth=15):
        engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)
        info = engine.analyse(board, limit=chess.engine.Limit(depth=depth))
        w_eval = info["score"].white().score(mate_score=10000)
        engine.quit()
        return w_eval

    def _calculate_entropy(self, board):
        return math.log(len(list(board.legal_moves)) + 1)

    ###########################################################################
    #                  USER MOVE (SAN) CALCULATION + DOT ON PLOT             #
    ###########################################################################
    def on_user_move_calculate(self, notation_entry, before_canvas, after_canvas, grade_label, entropy_label):
        move_text = notation_entry.get().strip()
        if not move_text:
            return

        try:
            # 1) push the user's move onto self.user_board => user_after
            user_move = self.user_board.parse_san(move_text)
            user_after = self.user_board.copy()
            user_after.push(user_move)

            # 2) Draw the after board
            self.draw_board_on_canvas(after_canvas, user_after)

            # 3) Evaluate user grade (white eval of the resulting position),
            #    compare to the best move, etc.
            move_eval, best_eval = self._compare_real_vs_best(self.user_board, user_move)
            user_grade = move_eval - best_eval

            # 4) Entropy from the AFTER position of the user move
            user_entropy = self._calculate_entropy(user_after)

            grade_label.config(text=f"{user_grade:.2f}")
            entropy_label.config(text=f"{user_entropy:.2f}")

            # 5) Add a red dot on the "Entropy Trends" plot (purely for visualization)
            x_val = self.current_move_index + 1
            self.user_moves_entropy_data.append((x_val, user_entropy))

            # 6) IMPORTANT: Store this user evaluation in the DataFrame
            #    and recalc the dependent columns for the current row (self.current_move_index).
            idx = self.current_move_index
            
            # The user’s evaluation from White’s perspective is "move_eval"
            self.evals_df.at[idx, "G_Friend"] = move_eval
            
            # Recalculate deviations with the new G_Friend
            self.evals_df.at[idx, "D_Actual_Friend"] = abs(self.evals_df.at[idx, "G_Actual"] - move_eval)
            self.evals_df.at[idx, "D_AI_Friend"]     = abs(self.evals_df.at[idx, "G_AI"]     - move_eval)

            # Chaos Index (std dev of [G_Actual, G_AI, G_Friend])
            vals = [
                self.evals_df.at[idx, "G_Actual"],
                self.evals_df.at[idx, "G_AI"],
                move_eval
            ]
            self.evals_df.at[idx, "ChaosIndex"] = np.std(vals)

            # Best vs. Played Outcome from the user’s perspective
            self.evals_df.at[idx, "Delta_Friend"] = self.evals_df.at[idx, "G_AI"] - move_eval

            # 7) Re-plot if "Entropy Trends" or "Correlation Analysis" is selected, etc.
            if self.plot_dropdown.get() != "Select Plot":
                self.update_plot(None)

        except Exception as e:
            notation_entry.delete(0, tk.END)
            notation_entry.insert(0, "Invalid move")


    ###########################################################################
    #                          PLOTTING METHODS                               #
    ###########################################################################
    def update_plot(self, event):
        plot_type = self.plot_dropdown.get()
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        if plot_type == "Evaluation Over Time":
            ax.plot(self.evals_df.index + 1, self.evals_df["eval_white"], label="White Eval")
            ax.plot(self.evals_df.index + 1, self.evals_df["eval_black"], label="Black Eval")
            ax.set_title("Evaluation Over Time")
            ax.set_xlabel("Move Number")
            ax.set_ylabel("Eval (White/Black)")
            ax.legend()

        elif plot_type == "Entropy Trends":
            # Original game entropies
            ax.plot(self.evals_df.index + 1, self.evals_df["entropy"], label="Entropy")
            ax.set_title("Entropy Over Moves")
            ax.set_xlabel("Move Number")
            ax.set_ylabel("ln(#legal_moves + 1)")

            # If the user has made any custom moves, show them as red dots
            if self.user_moves_entropy_data:
                xs, ys = zip(*self.user_moves_entropy_data)
                ax.scatter(xs, ys, color='red', marker='o', label="User Move Entropy")

            ax.legend()

        elif plot_type == "Deviation Maps":
            # Heatmap of [D_Actual_AI, D_Actual_Friend, D_AI_Friend]
            heat_data = self.evals_df[["D_Actual_AI", "D_Actual_Friend", "D_AI_Friend"]]
            # We transpose so that each of the three columns becomes a row in the heatmap
            sns.heatmap(heat_data.T, ax=ax, cmap="viridis", annot=True, cbar=True)
            ax.set_title("Deviation Maps (Absolute Differences)")
            ax.set_xlabel("Move Index")
            ax.set_ylabel("Deviation Type")

        elif plot_type == "Chaos Index":
            ax.plot(self.evals_df.index + 1, self.evals_df["ChaosIndex"], label="ChaosIndex", color='purple')
            ax.set_title("Chaos Index Over Moves")
            ax.set_xlabel("Move Number")
            ax.set_ylabel("Std Dev of [G_Actual, G_AI, G_Friend]")
            ax.legend()

        elif plot_type == "Best vs. Played Outcome":
            ax.plot(self.evals_df.index + 1, self.evals_df["Delta_AI"], label="Δ AI = G_AI - G_Actual", color='blue')
            ax.plot(self.evals_df.index + 1, self.evals_df["Delta_Friend"], label="Δ Friend = G_AI - G_Friend", color='green')
            ax.set_title("Best vs. Played Outcome")
            ax.set_xlabel("Move Number")
            ax.set_ylabel("Difference")
            ax.legend()

        elif plot_type == "Chess Chaos Art":
            # Scatter plot: x=move number, y=ChaosIndex, color by D_Actual_AI
            scatter = ax.scatter(
                x=self.evals_df.index + 1,
                y=self.evals_df["ChaosIndex"],
                c=self.evals_df["D_Actual_AI"],
                cmap="viridis"
            )
            ax.set_title("Chess Chaos Art")
            ax.set_xlabel("Move Number")
            ax.set_ylabel("ChaosIndex")
            cbar = self.figure.colorbar(scatter, ax=ax)
            cbar.set_label("D_Actual_AI")

        elif plot_type == "Correlation Analysis":
            # Correlation among [G_Actual, G_AI, G_Friend, entropy, ChaosIndex]
            corr_cols = ["G_Actual", "G_AI", "G_Friend", "entropy", "ChaosIndex"]
            corr_df = self.evals_df[corr_cols].dropna()
            if corr_df.empty:
                ax.text(0.5, 0.5, "No user moves entered yet\n(or no valid data).",
                ha='center', va='center', fontsize=12)
            else:
                corr_matrix = corr_df.corr()
                sns.heatmap(corr_matrix, ax=ax, annot=True, cmap="viridis", cbar=True)
                ax.set_title("Correlation Analysis")

        self.figure.tight_layout()
        self.mpl_canvas.draw()

###############################################################################
#                                 MAIN DEMO                                   #
###############################################################################

def main():
    """
    Demonstration of how to load multiple games from PGN, process them,
    and visualize in the combined UI (scrollable + user move input + multi-game).
    We add new metrics/plots:
      - Deviation Maps
      - Chaos Index
      - Best vs. Played Outcome
      - Chess Chaos Art
      - Correlation Analysis
    """
    # Adjust paths to match your environment
    pgn_file_path = "Dataset/lichess_decompressed.pgn"
    stockfish_path = "stockfish/stockfish-windows-x86-64-avx2.exe"

    # Load up to 5 games
    games = load_pgn_games(pgn_file_path, max_games=25)
    if not games:
        print("No games found in PGN!")
        return

    # Launch the Tkinter app
    root = tk.Tk()
    app = ChessMoveEvaluator(root, games, stockfish_path)
    root.mainloop()

if __name__ == "__main__":
    main()