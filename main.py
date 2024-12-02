import chess
import chess.pgn
import chess.engine
import pandas as pd
import os
import matplotlib.pyplot as plt
import tkinter as tk
from PIL import Image, ImageTk
import threading
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from chess_visualizer import ChessGUI

# Paths
pgn_file_path = 'Dataset/lichess_decompressed.pgn'
stockfish_path = 'stockfish/stockfish-windows-x86-64-avx2.exe'

# Step 1: Data Collection and Processing
def load_pgn_games(pgn_path, max_games=1000):
    """Load games from PGN file and return a list of games."""
    games = []
    with open(pgn_path, "r", encoding="utf-8") as pgn:
        for _ in range(max_games):
            game = chess.pgn.read_game(pgn)
            if game is None:
                break
            games.append(game)
    return games

def evaluate_game_with_stockfish(game, stockfish_path):
    """Evaluate a game using Stockfish and return a dataframe of moves and evaluations."""
    evaluations = []
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)

    # Create a board from the starting FEN of the game
    board = game.board()

    for move in game.mainline_moves():
        try:
            # Push the move to the board
            board.push(move)
            # Perform the analysis
            info = engine.analyse(board, chess.engine.Limit(depth=20))
            evaluations.append({
                "move": move.uci(),  # Get the move in UCI notation
                "evaluation_white": info["score"].white().score(mate_score=10000),  # Evaluation for White
                "evaluation_black": info["score"].black().score(mate_score=10000)   # Evaluation for Black
            })
        except Exception as e:
            print(f"Error processing move {move.uci()} in game: {game.headers.get('Event', 'Unknown')}, error: {e}")
            print(f"Board state:\n{board}")
            break

    engine.quit()
    return pd.DataFrame(evaluations)

def process_games(games, stockfish_path):
    """Process a list of games with Stockfish and return a consolidated dataset."""
    all_evaluations = []
    for i, game in enumerate(games):
        print(f"Processing game {i + 1}/{len(games)}")
        game_evaluations = evaluate_game_with_stockfish(game, stockfish_path)
        game_evaluations["game_id"] = i + 1
        game_evaluations["move_number"] = range(1, len(game_evaluations) + 1)
        all_evaluations.append(game_evaluations)
    return pd.concat(all_evaluations, ignore_index=True)

# Step 2: Visualization
def plot_zoom(event, ax, canvas):
    """Zoom in and out of the plot using mouse scroll."""
    # Get the current x and y limits
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Determine the zoom factor
    zoom_factor = 1.1 if event.delta > 0 else 0.9  # Scroll up to zoom in, scroll down to zoom out

    # Get the mouse position to center the zoom around that point
    widget = canvas.get_tk_widget()
    x_mouse = event.x
    y_mouse = event.y

    # Normalize mouse position relative to plot area
    ax_x_range = xlim[1] - xlim[0]
    ax_y_range = ylim[1] - ylim[0]
    x_frac = (x_mouse / widget.winfo_width()) * ax_x_range
    y_frac = (y_mouse / widget.winfo_height()) * ax_y_range

    # Apply the zoom factor
    new_xlim = [xlim[0] + x_frac * (1 - zoom_factor), xlim[1] - (ax_x_range - x_frac) * (1 - zoom_factor)]
    new_ylim = [ylim[0] + y_frac * (1 - zoom_factor), ylim[1] - (ax_y_range - y_frac) * (1 - zoom_factor)]

    # Set the new limits
    ax.set_xlim(new_xlim)
    ax.set_ylim(new_ylim)

    # Redraw the canvas to reflect the changes
    canvas.draw()

def on_press(event, ax, canvas):
    """Initiate dragging to pan the plot."""
    canvas.initial_x = event.x
    canvas.initial_y = event.y
    canvas.initial_xlim = ax.get_xlim()
    canvas.initial_ylim = ax.get_ylim()

def on_drag(event, ax, canvas):
    """Handle mouse drag to pan the plot."""
    # Calculate the change in mouse position (delta)
    dx = event.x - canvas.initial_x
    dy = event.y - canvas.initial_y

    # Get the current limits of the plot
    xlim = canvas.initial_xlim
    ylim = canvas.initial_ylim

    # Calculate the ranges of the x and y axes
    ax_x_range = xlim[1] - xlim[0]
    ax_y_range = ylim[1] - ylim[0]

    # Get the dimensions of the widget (canvas) holding the plot
    widget_width = canvas.get_tk_widget().winfo_width()
    widget_height = canvas.get_tk_widget().winfo_height()

    # Calculate the new x and y limits based on the mouse movement (panning)
    new_xlim = [xlim[0] - dx * ax_x_range / widget_width,
                xlim[1] - dx * ax_x_range / widget_width]
    new_ylim = [ylim[0] + dy * ax_y_range / widget_height,
                ylim[1] + dy * ax_y_range / widget_height]

    # Update the plot limits
    ax.set_xlim(new_xlim)
    ax.set_ylim(new_ylim)

    # Redraw the canvas to reflect the changes
    canvas.draw()

def on_release(event, ax, canvas):
    """Handle mouse release to stop dragging."""
    # You could use this to finalize the view or reset any values if needed
    pass

def setup_dragging(ax, canvas):
    """Bind mouse events for dragging functionality."""
    # Bind mouse press and release events
    canvas.get_tk_widget().bind("<ButtonPress-1>", lambda event: on_press(event, ax, canvas))
    canvas.get_tk_widget().bind("<B1-Motion>", lambda event: on_drag(event, ax, canvas))
    canvas.get_tk_widget().bind("<ButtonRelease-1>", lambda event: on_release(event, ax, canvas))

def plot_evaluations(evaluations, game_id, chess_gui):
    """Plot evaluations and delta evaluations for White and Black."""
    evaluations["delta_evaluation_white"] = evaluations["evaluation_white"].diff().fillna(0)
    evaluations["delta_evaluation_black"] = evaluations["evaluation_black"].diff().fillna(0)

    game_data = evaluations[evaluations["game_id"] == game_id]
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(game_data["move_number"], game_data["evaluation_white"], label="White Evaluation")
    ax.plot(game_data["move_number"], game_data["evaluation_black"], label="Black Evaluation")
    ax.plot(game_data["move_number"], game_data["delta_evaluation_white"], label="Delta White Evaluation", linestyle="--")
    ax.plot(game_data["move_number"], game_data["delta_evaluation_black"], label="Delta Black Evaluation", linestyle="--")
    ax.set_xlabel("Move Number")
    ax.set_ylabel("Evaluation")
    ax.set_title(f"Game {game_id} - Evaluations and Deltas")
    ax.legend()
    ax.grid()

    # Embed the plot in Tkinter
    canvas = FigureCanvasTkAgg(fig, master=chess_gui.root)
    canvas.draw()
    canvas.get_tk_widget().pack()

    canvas.get_tk_widget().bind("<MouseWheel>", lambda event, ax=ax, canvas=canvas: plot_zoom(event, ax, canvas))

    canvas.get_tk_widget().bind("<ButtonPress-1>", lambda event: on_press(event, ax, canvas))
    canvas.get_tk_widget().bind("<B1-Motion>", lambda event: on_drag(event, ax, canvas))
    canvas.get_tk_widget().bind("<ButtonRelease-1>", lambda event: on_release(event, ax, canvas))


    return canvas

def visualize_game(chess_gui, game):
    """Visualize the game move by move using Tkinter GUI."""
    for move in game.mainline_moves():
        chess_gui.next_move()  # Apply the move to the board
        chess_gui.update_board()  # Update the board display

# Step 3: Save Processed Data
def save_evaluations(evaluations, output_path):
    """Save evaluations to a CSV file."""
    if os.path.exists(output_path):
        open(output_path, 'w').close() 

    evaluations.to_csv(output_path, index=False)

# Main Script
def main():
    # Load PGN games
    print("Loading games...")
    games = load_pgn_games(pgn_file_path, max_games=3)  # Adjust to load more games if necessary

    # Process games with Stockfish
    print("Evaluating games...")
    evaluations = process_games(games, stockfish_path)

    # Save evaluations to a CSV file
    output_path = "chess_evaluations.csv"
    print(f"Saving evaluations to {output_path}...")
    save_evaluations(evaluations, output_path)

    # Visualize each game in sequence
    for game_id, game in enumerate(games, 1):
        print(f"Visualizing game {game_id}...")

        # Initialize Tkinter GUI
        chess_gui = ChessGUI(tk.Tk(), game)

        # Run the plotting function and update chessboard in the Tkinter window
        def close_all():
            plt.close()  # Close the plot
            chess_gui.root.quit()  # Close the Tkinter window
            chess_gui.root.destroy()  # Destroy the window completely to avoid re-use of destroyed components

        # Hook the window close event to close both the chessboard and plot
        chess_gui.root.protocol("WM_DELETE_WINDOW", close_all)

        # Run the plotting and chessboard visualization
        canvas = plot_evaluations(evaluations, game_id, chess_gui)

        # Run the Tkinter GUI in the main thread
        chess_gui.root.mainloop()

        # Now that the game visualization is finished, check if there's another game to visualize
        print(f"Game {game_id} finished.")

if __name__ == "__main__":
    main()
