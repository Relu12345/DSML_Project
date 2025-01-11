import math
import os
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import chess
import chess.pgn
import chess.engine
import pandas as pd

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
    Shows:
      - A scrollable left area with 3 rows:
          1) Real Game Move
          2) Stockfish Best Move
          3) Your Move (with a text field for the user's SAN move)
        Each row has Before/After boards plus Grade & Entropy labels to the right.
      - On the right side, a Plot dropdown and a Matplotlib figure
      - Navigation buttons for Next/Prev (to move through the moves)
      - Also "Prev Game" / "Next Game" to switch among multiple games
      - A title label for the game (White vs Black, date/time)
    """

    def __init__(self, root, games, stockfish_path):
        """
        :param root: the Tk root
        :param games: a list of chess.pgn.Game
        :param stockfish_path: path to Stockfish engine
        """
        self.root = root
        self.root.title("Chess Move Evaluator")
        # 1200 wide so we have enough space on the right
        self.root.geometry("1200x900")

        self.games = games
        self.stockfish_path = stockfish_path
        self.current_game_index = 0

        # We'll precompute evaluations for each game
        self.game_data = []  # list of (game, df_eval)

        for gm in self.games:
            df = self._evaluate_game_and_build_df(gm)
            self.game_data.append((gm, df))

        # UI top frame for "Prev Game" / "Next Game" and game title
        top_frame = tk.Frame(self.root)
        top_frame.pack(side=tk.TOP, fill=tk.X, pady=5)

        self.prev_game_btn = tk.Button(top_frame, text="<< Prev Game", command=self.prev_game)
        self.prev_game_btn.pack(side=tk.LEFT, padx=5)

        self.next_game_btn = tk.Button(top_frame, text="Next Game >>", command=self.next_game)
        self.next_game_btn.pack(side=tk.LEFT, padx=5)

        self.game_title_label = tk.Label(top_frame, text="Game Title", font=("Arial", 14, "bold"))
        self.game_title_label.pack(side=tk.LEFT, padx=20)

        # The main frames: left = boards, right = plot
        self.scroll_frame = ScrollableFrame(self.root)
        self.scroll_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        right_frame = tk.Frame(self.root, padx=10, pady=10)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # The three rows inside scroll_frame
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

        # On the right: Plot dropdown & Matplotlib figure
        tk.Label(right_frame, text="Select Plot:", font=("Arial", 14)).grid(row=0, column=0, sticky="w")
        self.plot_dropdown = ttk.Combobox(right_frame, 
                                          values=["Evaluation Over Time", "Entropy Trends"], 
                                          state="readonly")
        self.plot_dropdown.grid(row=0, column=1, sticky="w")
        self.plot_dropdown.set("Select Plot")
        self.plot_dropdown.bind("<<ComboboxSelected>>", self.update_plot)

        self.figure = plt.Figure(figsize=(5, 4), dpi=100)
        self.mpl_canvas = FigureCanvasTkAgg(self.figure, master=right_frame)
        self.mpl_canvas.get_tk_widget().grid(row=1, column=0, columnspan=2, pady=10)

        # Navigation for moves
        nav_frame = tk.Frame(right_frame)
        nav_frame.grid(row=2, column=0, columnspan=2, pady=15)
        tk.Button(nav_frame, text="<< Prev Move", command=self.prev_move).pack(side=tk.LEFT, padx=5)
        tk.Button(nav_frame, text="Next Move >>", command=self.next_move).pack(side=tk.LEFT, padx=5)

        # Start by loading the first game
        self.load_game(self.current_game_index)

    def _evaluate_game_and_build_df(self, game):
        """
        Evaluate 'game' with stockfish, compute a DataFrame with evals + entropies.
        """
        evals = evaluate_game_with_stockfish(game, self.stockfish_path, depth=15)
        entropies = calculate_entropy_for_game(game)
        df = pd.DataFrame(evals)
        df["entropy"] = entropies
        return df

    def load_game(self, idx):
        """
        Load the game at index 'idx' into the UI.
        We reset the move index to -1 (start of game).
        """
        if idx < 0 or idx >= len(self.games):
            return

        self.current_game_index = idx
        self.game, self.evals_df = self.game_data[idx]
        self.all_moves = list(self.game.mainline_moves())
        self.current_move_index = 0

        # Show a nice title
        self.show_game_title()

        # Initialize boards
        self.initialize_chess_boards()
        # Then do an initial update (so user sees the start position)
        self.update_all_boards()

    def prev_game(self):
        if self.current_game_index > 0:
            self.load_game(self.current_game_index - 1)

    def next_game(self):
        if self.current_game_index < len(self.games) - 1:
            self.load_game(self.current_game_index + 1)

    def show_game_title(self):
        """
        Show something like: "WhiteName vs BlackName - 2023.04.01 13:00:00"
        based on PGN headers, if they exist.
        """
        headers = self.game.headers
        white = headers.get("White", "Unknown")
        black = headers.get("Black", "Unknown")
        date = headers.get("UTCDate", "")
        time = headers.get("UTCTime", "")
        title_text = f"{white} vs {black} - {date} {time}"

        self.game_title_label.config(text=title_text)

    ###########################################################################
    #                          ROW CREATION / UI                              #
    ###########################################################################
    def add_move_section(self, parent, label_text, show_user_input=False):
        """
        Creates a row with:
          - A label (e.g., "Real Game Move")
          - Before canvas, After canvas
          - Grade label, Entropy label to the right
          - If show_user_input=True, add a text entry & button below for SAN moves
        Returns a dictionary of the created widgets.
        """
        section_frame = tk.Frame(parent, pady=10)
        section_frame.pack(fill=tk.X, expand=True)

        # Title
        tk.Label(section_frame, text=label_text, font=("Arial", 16)).pack(anchor="w")

        # Row for boards + grade/entropy
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
        """
        Display placeholder text on boards (before any moves).
        """
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
        canvas.image = img_tk  # keep ref

    ###########################################################################
    #                 NEXT/PREV MOVE + BOARD UPDATING LOGIC                   #
    ###########################################################################

    def next_move(self):
        if self.current_move_index < len(self.all_moves) - 1:
            self.current_move_index += 1
            self.update_all_boards()

    def prev_move(self):
        if self.current_move_index >= 0:
            self.current_move_index -= 1
            self.update_all_boards()

    def update_all_boards(self):
        """
        Rebuild boards for:
          1) Real Game Move
          2) Stockfish Best Move
          3) "Your Move"
        Then update Grade & Entropy for each row.
        """

        # --------------------------
        # REAL GAME MOVE
        # --------------------------
        real_before_board = self.game.board()
        for m in self.all_moves[:self.current_move_index]:
            real_before_board.push(m)


        real_after_board = real_before_board.copy()
        real_move = None
        if self.current_move_index >= 0:
            real_move = self.all_moves[self.current_move_index]
            real_after_board.push(real_move)

        # Draw boards
        self.draw_board_on_canvas(self.real_move_row["before_canvas"], real_before_board)
        self.draw_board_on_canvas(self.real_move_row["after_canvas"], real_after_board)

        # Evaluate real move vs best
        real_eval, best_eval = self._compare_real_vs_best(real_before_board, real_move)
        # Grade for Real row = (real_eval - best_eval)
        real_grade = real_eval - best_eval
        real_entropy = self._calculate_entropy(real_before_board)
        self.real_move_row["grade_label"].config(text=f"{real_grade:.2f}")
        self.real_move_row["entropy_label"].config(text=f"{real_entropy:.2f}")

        # --------------------------
        # STOCKFISH BEST MOVE
        # --------------------------
        sf_before_board = real_before_board.copy()
        sf_best_move = get_stockfish_best_move(sf_before_board, self.stockfish_path) if sf_before_board is not None else None
        sf_after_board = sf_before_board.copy()
        if sf_best_move is not None:
            sf_after_board.push(sf_best_move)

        self.draw_board_on_canvas(self.sf_move_row["before_canvas"], sf_before_board)
        self.draw_board_on_canvas(self.sf_move_row["after_canvas"], sf_after_board)

        # Evaluate best move vs real
        # (Now best_eval - real_eval for "Stockfish row" to show how much better the best move is)
        sf_eval, best_eval_2 = self._compare_real_vs_best(sf_before_board, sf_best_move)
        # But note _compare_real_vs_best returns (eval_of_that_move, eval_of_best_move).
        # If 'that_move' is the best move, then "that_move_eval" ~ best_eval, 
        #  and "eval_of_best_move" is the best-of-the-best (which might be the same move).
        # We'll do a simpler approach: we also evaluate the real move from the same position:
        # Then "Stockfish grade" = (best_eval - real_eval).
        real_eval_from_sfpos, best_eval_from_sfpos = self._compare_real_vs_best(sf_before_board, real_move)

        # Actually, we want the real eval from the same position as 'sf_before_board'.
        # If 'real_move' is not possible from 'sf_before_board' in the same state, we do a quick approach:
        if real_move is not None:
            temp2 = sf_before_board.copy()
            if real_move in temp2.legal_moves:
                temp2.push(real_move)
                real_eval_sfpos = self._evaluate_position(temp2)
            else:
                # If the real move can't be played from this position, just 0
                real_eval_sfpos = 0.0
        else:
            real_eval_sfpos = 0.0

        # Evaluate the SF move
        if sf_best_move is not None:
            temp3 = sf_before_board.copy()
            temp3.push(sf_best_move)
            best_eval_sfpos = self._evaluate_position(temp3)
        else:
            best_eval_sfpos = 0.0

        # Now Stockfish row "Grade" = (best_eval - real_eval)
        # from the same position. 
        sf_grade = best_eval_sfpos - real_eval_sfpos
        sf_entropy = self._calculate_entropy(sf_before_board)
        self.sf_move_row["grade_label"].config(text=f"{sf_grade:.2f}")
        self.sf_move_row["entropy_label"].config(text=f"{sf_entropy:.2f}")

        # --------------------------
        # USER MOVE
        # --------------------------
        # Start user board from the real_before_board
        self.user_board = real_before_board.copy()

        user_after_board = self.user_board.copy()  # no user move yet
        self.draw_board_on_canvas(self.user_move_row["before_canvas"], self.user_board)
        self.draw_board_on_canvas(self.user_move_row["after_canvas"], user_after_board)

        # Reset user row grade/entropy
        self.user_move_row["grade_label"].config(text="0.00")
        self.user_move_row["entropy_label"].config(text="0.00")

        # If a plot is selected, update it
        if self.plot_dropdown.get() != "Select Plot":
            self.update_plot(None)

    ###########################################################################
    #                      REAL vs BEST EVALUATION HELPERS                    #
    ###########################################################################
    def _compare_real_vs_best(self, board_before, move):
        """
        Evaluate 'move' from 'board_before' (return that_move_eval) 
        and also evaluate the best move from board_before (return best_move_eval).
        Returns (that_move_eval, best_move_eval) from White's perspective.
        """
        # Evaluate the move in question
        if move is None:
            that_move_eval = 0.0
        else:
            temp_board = board_before.copy()
            if move in temp_board.legal_moves:
                temp_board.push(move)
                that_move_eval = self._evaluate_position(temp_board)
            else:
                # If the move is not legal from that position, 0
                that_move_eval = 0.0

        # Evaluate the best move
        best_move = get_stockfish_best_move(board_before, self.stockfish_path)
        best_eval = 0.0
        if best_move is not None:
            temp2 = board_before.copy()
            temp2.push(best_move)
            best_eval = self._evaluate_position(temp2)

        return (that_move_eval, best_eval)

    def _evaluate_position(self, board, depth=15):
        """
        Evaluate position with Stockfish, return White's numeric eval.
        """
        engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)
        info = engine.analyse(board, limit=chess.engine.Limit(depth=depth))
        w_eval = info["score"].white().score(mate_score=10000)
        engine.quit()
        return w_eval

    def _calculate_entropy(self, board):
        return math.log(len(list(board.legal_moves)) + 1)

    ###########################################################################
    #                  USER MOVE CALCULATION (SAN Input)                      #
    ###########################################################################
    def on_user_move_calculate(self, notation_entry, before_canvas, after_canvas, grade_label, entropy_label):
        """
        Called when the user clicks "Calculate" in the "Your Move" row.
        - We parse the SAN move from the user input
        - We push it onto self.user_board
        - We update the "after" board, and recalc Grade & Entropy
        """
        move_text = notation_entry.get().strip()
        if not move_text:
            return

        try:
            user_move = self.user_board.parse_san(move_text)
            # BEFORE board is self.user_board
            user_after = self.user_board.copy()
            user_after.push(user_move)

            # Draw the after board
            self.draw_board_on_canvas(after_canvas, user_after)

            # Evaluate user's move vs best
            user_eval, best_eval = self._compare_real_vs_best(self.user_board, user_move)
            user_grade = user_eval - best_eval
            user_entropy = self._calculate_entropy(self.user_board)

            grade_label.config(text=f"{user_grade:.2f}")
            entropy_label.config(text=f"{user_entropy:.2f}")

        except Exception as e:
            notation_entry.delete(0, tk.END)
            notation_entry.insert(0, "Invalid move")

    ###########################################################################
    #                           PLOTTING METHODS                              #
    ###########################################################################
    def update_plot(self, event):
        """
        Draw the chosen plot (Evaluation Over Time or Entropy Trends)
        from self.evals_df for the current game.
        """
        plot_type = self.plot_dropdown.get()
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        if plot_type == "Evaluation Over Time":
            ax.plot(self.evals_df.index + 1, self.evals_df["eval_white"], label="White Eval")
            ax.plot(self.evals_df.index + 1, self.evals_df["eval_black"], label="Black Eval")
            ax.set_title("Evaluation Over Time")
            ax.set_xlabel("Move Number")
            ax.set_ylabel("Eval (White/Black)")
        elif plot_type == "Entropy Trends":
            ax.plot(self.evals_df.index + 1, self.evals_df["entropy"], label="Entropy")
            ax.set_title("Entropy Over Moves")
            ax.set_xlabel("Move Number")
            ax.set_ylabel("ln(#legal_moves + 1)")

        ax.legend()
        self.mpl_canvas.draw()

###############################################################################
#                                 MAIN DEMO                                   #
###############################################################################

def main():
    """
    Demonstration of how to load multiple games from PGN, process them,
    and visualize in the combined UI (scrollable + user move input + multi-game).
    """
    # Adjust paths to match your environment
    pgn_file_path = "Dataset/lichess_decompressed.pgn"
    stockfish_path = "stockfish/stockfish-windows-x86-64-avx2.exe"

    # Load up to 2 games (or more)
    games = load_pgn_games(pgn_file_path, max_games=5)
    if not games:
        print("No games found in PGN!")
        return

    # Launch the Tkinter app
    root = tk.Tk()
    app = ChessMoveEvaluator(root, games, stockfish_path)
    root.mainloop()

if __name__ == "__main__":
    main()
