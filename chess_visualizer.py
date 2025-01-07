import os
import tkinter as tk
from PIL import Image, ImageTk
import chess

class ChessGUI:
    def __init__(self, root, game):
        self.root = root
        self.game = game
        self.board = game.board()  # Initialize with the starting position of the game
        self.moves = list(game.mainline_moves())  # List of all moves in the game
        self.move_index = 0  # Index to keep track of the current move

        # Set up the window
        self.root.title("Chess Game Viewer")
        self.root.geometry("1024x950")

        # Create a frame for the control buttons and current move label
        control_frame = tk.Frame(self.root)
        control_frame.pack(side=tk.TOP, padx=10, pady=10)

        # Create control buttons
        self.prev_button = tk.Button(control_frame, text="<< Prev", command=self.prev_move)
        self.prev_button.pack(side=tk.LEFT, padx=10)

        self.next_button = tk.Button(control_frame, text="Next >>", command=self.next_move)
        self.next_button.pack(side=tk.LEFT, padx=10)

        self.best_move_button = tk.Button(control_frame, text="Show Best Move", command=self.show_best_move)
        self.best_move_button.pack(side=tk.LEFT, padx=10)

        # Create the current move label
        self.current_move_label = tk.Label(control_frame, text="Start of game", font=('Arial', 14))
        self.current_move_label.pack(side=tk.LEFT, padx=10)

        # Create the chessboard canvas
        self.canvas = tk.Canvas(self.root, width=520, height=520)  # Adjust size for 60x60 squares
        self.canvas.pack(pady=10)

        # Initial display
        self.update_board()

    def update_board(self):
        """Update the board display on the canvas."""
        self.canvas.delete("all")  # Clear previous pieces

        # Get the board image after applying the current moves
        img = self.render_board_image(self.board)

        # Convert the image to a Tkinter-compatible photo image
        img_tk = ImageTk.PhotoImage(img)

        # Display the image on the Tkinter canvas
        self.canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)

        # Keep a reference to the image to avoid garbage collection
        self.canvas.image = img_tk

        # Update the current move label
        self.update_current_move()

    def next_move(self):
        """Move to the next move in the game."""
        if self.move_index < len(self.moves):
            move = self.moves[self.move_index]
            self.board.push(move)  # Apply the move to the board
            self.move_index += 1  # Move forward in the game
            self.update_board()  # Update the display

    def prev_move(self):
        """Move to the previous move in the game."""
        if self.move_index > 0:
            self.move_index -= 1  # Move backward in the game
            self.board.pop()  # Revert the last move
            self.update_board()  # Update the display

    def update_current_move(self):
        """Update the current move label with the move details."""
        if self.move_index == 0:
            self.current_move_label.config(text="Start of game")
        else:
            # Create a temporary board to get the correct move notation
            temp_board = chess.Board()
            # Apply all moves up to the current index
            for i in range(self.move_index - 1):
                temp_board.push(self.moves[i])
            
            move = self.moves[self.move_index - 1]
            move_san = temp_board.san(move)  # Get move in algebraic notation
            player = "White" if (self.move_index % 2 == 1) else "Black"
            self.current_move_label.config(text=f"Move {self.move_index}: {player} ({move_san})")

    def get_best_move(self):
        """Get Stockfish's best move recommendation for current position"""
        engine = chess.engine.SimpleEngine.popen_uci("stockfish/stockfish-windows-x86-64-avx2.exe")
        result = engine.play(self.board, chess.engine.Limit(time=2.0))
        best_move = result.move
        engine.quit()
        return best_move

    def show_best_move(self):
        """Display the best move in the current position"""
        best_move = self.get_best_move()
        # Convert move to algebraic notation for display
        best_move_san = self.board.san(best_move)
        self.current_move_label.config(text=f"Best move: {best_move_san}")

    def render_board_image(self, board):
        """Render the chessboard to an image with notation labels."""
        square_size = 60
        margin = 20  # Space for labels
        total_size = 8 * square_size + 2 * margin
        img = Image.new("RGBA", (total_size, total_size), color=(255, 255, 255, 0))

        # Draw the chessboard squares
        for row in range(8):
            for col in range(8):
                color = (255, 255, 255) if (row + col) % 2 == 0 else (169, 169, 169)
                for i in range(square_size):
                    for j in range(square_size):
                        img.putpixel((col * square_size + margin + i, row * square_size + margin + j), color)

        # Draw the pieces
        for row in range(8):
            for col in range(8):
                piece = board.piece_at(chess.square(col, 7 - row))
                if piece:
                    piece_image = self.load_piece_image(piece.symbol())
                    if piece_image:
                        img.paste(piece_image, (col * square_size + margin, row * square_size + margin), piece_image)

        # Create a drawing context
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        
        # Add file letters (a-h)
        for col in range(8):
            letter = chr(ord('a') + col)
            # Bottom letters
            draw.text((col * square_size + margin + 25, total_size - margin + 5), 
                    letter, fill=(0, 0, 0))
            # Top letters
            draw.text((col * square_size + margin + 25, 5), 
                    letter, fill=(0, 0, 0))

        # Add rank numbers (1-8)
        for row in range(8):
            number = str(8 - row)
            # Left numbers
            draw.text((5, row * square_size + margin + 25), 
                    number, fill=(0, 0, 0))
            # Right numbers
            draw.text((total_size - margin + 5, row * square_size + margin + 25), 
                    number, fill=(0, 0, 0))

        return img

    def load_piece_image(self, symbol):
        """Load an image for the chess piece."""
        pieces = {'p': 'pawn', 'r': 'rook', 'n': 'knight', 'b': 'bishop', 'q': 'queen', 'k': 'king'}
        
        piece = pieces.get(symbol.lower())
        color = 'white' if symbol.isupper() else 'black'  # Determine color based on case
        
        if piece and color:
            img_path = f"images/{color}_{piece}.png"  # Ensure images are named like 'white_pawn.png'
            if os.path.exists(img_path):
                try:
                    # Load image with alpha channel (transparency)
                    img = Image.open(img_path).convert("RGBA")  # Convert to RGBA to handle transparency
                    img = img.resize((60, 60), Image.Resampling.LANCZOS)  # Resize image
                    return img
                except Exception as e:
                    print(f"Error loading image for {color}_{piece}: {e}")
            else:
                print(f"Image path does not exist: {img_path}")
        
        return None  # Return None if image could not be loaded
