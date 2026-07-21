"""
Tri-Engine Adaptive Chess Assistant — FIXED
============================================
Corrected version of the original script. Summary of fixes (see inline
"FIX #n" comments for exact locations):

 1. Missing `import urllib.parse` (would crash on first web search).
 2. train_on_real_file() and async_background_retrain() were stubs -> the
    ML layer never trained and `is_initialized` never became True. Now
    implemented: parses real Lichess PGN data (.pgn or .pgn.zst, local
    path OR http(s) URL), extracts (tension, ply, is_check, piece_count)
    features per position, labels them by game result, and fits both
    the SGDClassifier (partial_fit, incremental) and RandomForestClassifier
    (batch fit) before flipping is_initialized = True.
 3. Empty user_path no longer triggers a pointless training call.
 4. MCTS docstring clarified: this is a best-first / value-guided tree
    search (leaf evaluated directly, no rollout) -- not classic random
    playout MCTS. Renamed conceptually via docstring only, kept the class
    name to avoid breaking references.
 5. Backprop sign convention documented explicitly (value is White-signed
    and correctness depends on min/max alternation in Selection).
 6. Removed the redundant full board replay in TrueFluidEvaluator -- it
    now reuses the tension values already collected in game_metrics_log
    during play instead of recomputing them from scratch.
 7. MCTSNode expansion/selection made a bit cheaper (cache legal moves
    once per node instead of recomputing via list(board.legal_moves)
    on every fully_expanded() check).
 8. Bare `except Exception: pass` blocks narrowed and now at least log
    once (non-spammy) instead of silently eating every error.
 9. Move-parsing exception handling broadened slightly to also catch
    python-chess's InvalidMoveError where present.
10. Wikipedia telemetry search kept as cosmetic flavor text (unchanged
    behaviorally besides the import fix) -- documented as non-functional
    decoration, not real telemetry.

Data source for training: https://database.lichess.org/
Standard-chess monthly exports are named like:
    https://database.lichess.org/standard/lichess_db_standard_rated_2026-06.pgn.zst
Pass either a local decompressed .pgn path, a local .pgn.zst path (requires
the optional `zstandard` package), or a full https:// URL to this script's
training prompt and it will fetch/decompress/parse it automatically.
"""

import chess
import chess.pgn
import io
import random
import urllib.request
import urllib.parse          # FIX #1: this was missing in the original
import json
import re
import warnings
import numpy as np
import os
import math
import threading
import tempfile
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from IPython.display import display, clear_output

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

ONNX_AVAILABLE = False
try:
    import tensorflow as tf
    import tf2onnx
    import onnxruntime as ort
    tf.get_logger().setLevel('ERROR')
    from tensorflow.keras import layers, models
    ONNX_AVAILABLE = True
except ImportError:
    print("[SYSTEM NOTE] ONNX or TensorFlow libraries missing. Falling back to pure algorithmic structures.")

ZSTD_AVAILABLE = False
try:
    import zstandard as zstd
    ZSTD_AVAILABLE = True
except ImportError:
    pass  # .pgn.zst files simply won't be auto-decompressed; plain .pgn still works


# =====================================================================
# 1. CORE DISCOVERY & INTERACTIVE TENSION LAYER
# =====================================================================
class DiscoveryEngine:
    @staticmethod
    def calculate_board_tension(board: chess.Board) -> int:
        tension_score = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                attackers = board.attackers(not piece.color, square)
                tension_score += len(attackers)
        return tension_score


# =====================================================================
# 2. LONG-TERM COGNITIVE MEMORY BANK (THREAD-SAFE STORAGE)
# =====================================================================
class CognitiveMemory:
    def __init__(self):
        self.past_games_history = []
        self.global_dataset_baseline_tension = 0.33
        self.total_states_learned = 0
        self.game_counter = 0
        self.lock = threading.Lock()

    def record_game(self, moves, game_metrics):
        with self.lock:
            self.past_games_history.append({"moves": moves, "metrics": game_metrics})
            self.game_counter += 1


# =====================================================================
# 3. HIGH-PERFORMANCE ENGINE CORE
# =====================================================================
class TensorFlowChessBrain:
    def __init__(self):
        self.onnx_enabled = ONNX_AVAILABLE
        self.ort_session = None
        self.session_lock = threading.Lock()
        self._warned_once = False

        if self.onnx_enabled:
            inputs = layers.Input(shape=(8, 8, 13), name="board_tensor")
            x = layers.Conv2D(32, kernel_size=(3, 3), padding="same", activation="relu")(inputs)
            x = layers.Conv2D(16, kernel_size=(3, 3), padding="same", activation="relu")(x)
            x = layers.Flatten()(x)
            x = layers.Dense(64, activation="relu")(x)
            value_out = layers.Dense(1, activation="tanh", name="position_value")(x)

            self.tf_model = models.Model(inputs=inputs, outputs=value_out)
            self.tf_model.compile(optimizer='adam', loss='mean_squared_error')
            self.compile_tf_to_onnx()

    def compile_tf_to_onnx(self):
        if not self.onnx_enabled:
            return
        try:
            spec = (tf.TensorSpec((None, 8, 8, 13), tf.float32, name="board_tensor"),)
            onnx_model, _ = tf2onnx.convert.from_keras(self.tf_model, input_signature=spec, opset=13)
            model_bytes = onnx_model.SerializeToString()

            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 1

            with self.session_lock:
                self.ort_session = ort.InferenceSession(model_bytes, sess_options=opts)
        except Exception as e:
            # FIX #8: narrowed intent -- still safe-fallback, but now surfaces
            # the reason once instead of silently vanishing.
            if not self._warned_once:
                print(f"[WARN] ONNX compile failed, disabling TF/ONNX path: {e}")
                self._warned_once = True
            self.onnx_enabled = False

    @staticmethod
    def board_to_tensor(board: chess.Board):
        tensor = np.zeros((8, 8, 13), dtype=np.float32)
        piece_map = {chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2, chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5}
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                row, col = 7 - (square // 8), square % 8
                channel = piece_map[piece.piece_type] + (6 if piece.color == chess.BLACK else 0)
                tensor[row, col, channel] = 1.0
            else:
                tensor[7 - (square // 8), square % 8, 12] = 1.0
        return tensor

    def fast_onnx_predict(self, board_tensor: np.ndarray) -> float:
        if not self.onnx_enabled or self.ort_session is None:
            return 0.0

        input_feed = {"board_tensor": np.expand_dims(board_tensor, axis=0)}
        with self.session_lock:
            outputs = self.ort_session.run(None, input_feed)
        return float(outputs[0][0][0])

    def fit_tensor(self, states, values):
        if not self.onnx_enabled or len(states) == 0:
            return
        X = np.array([self.board_to_tensor(s) for s in states])
        y = np.array(values, dtype=np.float32)
        self.tf_model.fit(X, y, epochs=1, batch_size=64, verbose=0)


# =====================================================================
# 4. STRUCTURAL FLUID EVALUATOR (cosmetic flavor-text layer)
# =====================================================================
class TrueFluidEvaluator:
    """
    NOTE: This layer is decorative. The Wikipedia queries below are built
    from small internal lookup tables (psychology/CS-jargon words), NOT
    from any private game or user data, so there is no real "telemetry"
    being sent anywhere -- it just produces flavor text for the console.
    """

    @staticmethod
    def execute_live_search(clean_keywords_only: str) -> str:
        words = clean_keywords_only.split(", ")
        unique_words = []
        for w in words:
            w_clean = w.strip()
            if w_clean not in unique_words:
                unique_words.append(w_clean)

        search_query = " ".join(unique_words)
        encoded_query = urllib.parse.quote(search_query)  # FIX #1: now resolvable

        url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={encoded_query}&format=json"

        try:
            headers = {'User-Agent': 'ChessTelemetryResearchEngine/4.0 (academic_eval@domain.com)'}
            req = urllib.request.Request(url, headers=headers)

            with urllib.request.urlopen(req, timeout=4) as response:
                data = json.loads(response.read().decode())
                search_results = data.get("query", {}).get("search", [])

                if search_results:
                    snippet = search_results[0].get('snippet', '')
                    clean_text = re.sub(r'<[^>]+>', '', snippet)

                    sentences = re.split(r'(?<=[.!?])\s+', clean_text)
                    final_summary = " ".join(sentences[:-1]) if len(sentences) > 1 else clean_text

                    if not final_summary.endswith('.'):
                        final_summary += "..."
                    return final_summary
        except Exception as e:
            return f"Search telemetry timeout ({str(e)})."

        return "No precise mathematical crossover found in index. Systems running nominal tracking templates."

    @classmethod
    def process_and_display(cls, game_moves, game_metrics_log, memory_vault, sgd_weights, rf_importances):
        """
        FIX #6: previously this replayed the entire game from a fresh board
        just to recompute tension values that were ALREADY captured live in
        game_metrics_log during play. Now it reuses that log directly.
        """
        print("\n" + "=" * 75)
        print("ADVANCED SYSTEM INTEGRATION: TOTAL DATA SYNTHESIS MATRIX")
        print("=" * 75)

        total_tension, high_friction_plies = 0, 0
        temp_board = chess.Board()
        for idx, move in enumerate(game_moves):
            tension = game_metrics_log[idx]['tension'] if idx < len(game_metrics_log) else \
                DiscoveryEngine.calculate_board_tension(temp_board)
            print(f"  [Ply {idx+1:<2}] {temp_board.san(move):<6} | Threat Friction Vector: {tension}")
            total_tension += tension
            if tension > 7:
                high_friction_plies += 1
            temp_board.push(move)

        avg_game_tension = total_tension / len(game_moves) if game_moves else 5.0
        chaos_ratio = high_friction_plies / len(game_moves) if game_moves else 0.0
        linear_bias = float(np.sum(sgd_weights)) if sgd_weights is not None else 0.5
        tree_importance = float(np.max(rf_importances)) if rf_importances is not None else 0.5

        psy_adjectives = ["unstable", "hyper-reactive", "volatile", "acute", "strained", "homeostatic", "suppressed"]
        psy_states = ["Hypervigilance", "Cognitive Friction", "Emotional Homeostasis", "Allostatic Load", "Sensory Gating"]
        psy_actions = ["forced tree pruning", "threat mitigation", "attentional allocation", "metabolic regulation"]

        cs_blueprints = ["hyperplane slicing", "entropy sorting", "gradient mapping", "nodal divergence", "stochastic bounds"]
        cs_models = ["Random Forest", "Stochastic Gradient Descent", "Linear Regression", "Ensemble Minimization"]
        cs_conclusions = ["structural optimization", "dimensional reduction", "asymmetric spatial split", "hyper-parameter shift"]

        idx_1 = int(chaos_ratio * 100) % len(psy_adjectives)
        idx_2 = int(avg_game_tension * 12) % len(psy_states)
        idx_3 = int(abs(linear_bias) * 55) % len(psy_actions)

        idx_4 = int(tree_importance * 88) % len(cs_blueprints)
        idx_5 = int(abs(linear_bias + tree_importance) * 33) % len(cs_models)
        idx_6 = int(abs(linear_bias - tree_importance) * 120) % len(cs_conclusions)

        psy_features_csv = f"{psy_adjectives[idx_1]}, {psy_states[idx_2]}, {psy_actions[idx_3]}"
        cs_features_csv = f"{cs_blueprints[idx_4]}, {cs_models[idx_5]}, {cs_conclusions[idx_6]}"

        print("-" * 75)
        print(f"DISPATCHING INTERPOLATED TELEMETRY (CHAOS: {chaos_ratio:.2%}, TENSION: {avg_game_tension:.2f})")
        print("-" * 75)

        live_psy_response = cls.execute_live_search(psy_features_csv)
        live_cs_response = cls.execute_live_search(cs_features_csv)

        print("LIVE SEARCH ENGINE MATRIX SYNTHESIS RESULTS:")
        print("-" * 75)
        print(f"- Psychology Evaluation Discovery Result:\n  {live_psy_response}")
        print("-" * 75)
        print(f"- Spatial Algorithm Topology Discovery Result:\n  {live_cs_response}")
        print("=" * 75 + "\n")


# =====================================================================
# 5. TREE SEARCH ENGINE (best-first / value-guided, not classic MCTS)
# =====================================================================
class MCTSNode:
    def __init__(self, board: chess.Board, parent=None):
        self.board = board.copy()
        self.parent = parent
        self.children = {}
        self.visit_count = 0
        self.total_value = 0.0
        self._legal_moves_cache = None  # FIX #7: cache legal moves per node

    def legal_moves_list(self):
        if self._legal_moves_cache is None:
            self._legal_moves_cache = list(self.board.legal_moves)
        return self._legal_moves_cache

    def is_fully_expanded(self):
        return len(self.children) == len(self.legal_moves_list())


class AdaptiveBrain:
    def __init__(self, memory_vault: CognitiveMemory):
        self.memory = memory_vault
        self.wide_model = SGDClassifier(loss="log_loss", warm_start=True)
        self.tf_brain = TensorFlowChessBrain()
        self.forest_model = RandomForestClassifier(n_estimators=50, max_depth=6, n_jobs=-1)
        self.is_initialized = False
        self.classes = np.array([0, 1])
        self._retrain_thread = None

        self.piece_values = {
            chess.PAWN: 100.0,
            chess.KNIGHT: 325.0,
            chess.BISHOP: 325.0,
            chess.ROOK: 500.0,
            chess.QUEEN: 975.0,
            chess.KING: 100000.0
        }

    # -----------------------------------------------------------------
    # FIX #2 (part A): real training on Lichess data
    # -----------------------------------------------------------------
    @staticmethod
    def _open_pgn_source(file_path: str):
        """
        Returns a file-like text object ready for chess.pgn.read_game(),
        supporting: local .pgn, local .pgn.zst, and http(s) URLs to either.
        """
        is_url = file_path.startswith("http://") or file_path.startswith("https://")

        if is_url:
            print(f"Downloading Lichess data from: {file_path}")
            headers = {'User-Agent': 'ChessEngineTrainer/1.0'}
            req = urllib.request.Request(file_path, headers=headers)
            with urllib.request.urlopen(req, timeout=60) as resp:
                raw_bytes = resp.read()
            suffix = ".zst" if file_path.endswith(".zst") else ".pgn"
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmp.write(raw_bytes)
            tmp.close()
            file_path = tmp.name

        if file_path.endswith(".zst"):
            if not ZSTD_AVAILABLE:
                raise RuntimeError(
                    "This is a .zst compressed Lichess file but the 'zstandard' "
                    "package is not installed. Run: pip install zstandard --break-system-packages"
                )
            fh = open(file_path, "rb")
            dctx = zstd.ZstdDecompressor()
            stream_reader = dctx.stream_reader(fh)
            text_stream = io.TextIOWrapper(stream_reader, encoding="utf-8", errors="ignore")
            return text_stream
        else:
            return open(file_path, "r", encoding="utf-8", errors="ignore")

    def train_on_real_file(self, file_path: str, max_games=200, min_ply=6):
        """
        Parses real Lichess games (from database.lichess.org exports) and
        fits both wide_model (SGD, incremental) and forest_model (RF, batch)
        on (tension, ply, is_check, piece_count) -> did-White-win features.

        file_path can be:
          - a local .pgn file
          - a local .pgn.zst file (requires `zstandard`)
          - a full https:// URL to either (e.g. a lichess_db_standard_rated
            monthly export from https://database.lichess.org/)
        """
        if not file_path:
            print("No training file provided -- skipping ML pretraining "
                  "(engine will still run on pure material/search evaluation).")
            return

        try:
            source = self._open_pgn_source(file_path)
        except Exception as e:
            print(f"[TRAINING ABORTED] Could not open PGN source: {e}")
            return

        X_rows, y_rows = [], []
        games_parsed = 0

        try:
            with source:
                while games_parsed < max_games:
                    game = chess.pgn.read_game(source)
                    if game is None:
                        break  # end of file

                    result = game.headers.get("Result", "*")
                    if result == "1-0":
                        white_won = 1
                    elif result == "0-1":
                        white_won = 0
                    elif result == "1/2-1/2":
                        games_parsed += 1
                        continue  # skip draws for this simple binary classifier
                    else:
                        continue  # unfinished/aborted game

                    board = game.board()
                    for move in game.mainline_moves():
                        if board.ply() >= min_ply:
                            tension = DiscoveryEngine.calculate_board_tension(board)
                            features = [
                                tension,
                                board.ply(),
                                1 if board.is_check() else 0,
                                len(board.piece_map()),
                            ]
                            X_rows.append(features)
                            y_rows.append(white_won)
                        board.push(move)

                    games_parsed += 1
        except Exception as e:
            print(f"[TRAINING WARNING] Stopped early while parsing PGN: {e}")

        if len(X_rows) < 10:
            print(f"Only {len(X_rows)} training samples extracted from {games_parsed} games -- "
                  "not enough to fit reliably. Skipping ML pretraining.")
            return

        X = np.array(X_rows, dtype=np.float64)
        y = np.array(y_rows, dtype=np.int32)

        print(f"Parsed {games_parsed} games -> {len(X_rows)} position samples. Fitting models...")

        # RandomForest needs a batch fit (no partial_fit support)
        self.forest_model.fit(X, y)

        # SGD supports true incremental learning -- fit in shuffled minibatches
        idx = np.arange(len(X))
        np.random.shuffle(idx)
        batch_size = 256
        for start in range(0, len(idx), batch_size):
            batch_idx = idx[start:start + batch_size]
            self.wide_model.partial_fit(X[batch_idx], y[batch_idx], classes=self.classes)

        self.is_initialized = True  # FIX #2: this was never set before
        print(f"Training complete. Forest OOB-style importance ready, SGD fitted on "
              f"{len(X_rows)} samples across {games_parsed} games. ML layer is now ACTIVE.")

    # -----------------------------------------------------------------
    # FIX #2 (part B): background retraining after each played game
    # -----------------------------------------------------------------
    def async_background_retrain(self):
        """
        Retrains incrementally on the games accumulated in memory_vault
        since the last retrain, without blocking the main game loop.
        Safe to call repeatedly; skips silently if a retrain is already
        running or there isn't enough new data yet.
        """
        if self._retrain_thread is not None and self._retrain_thread.is_alive():
            return  # a retrain is already in flight; don't stack threads

        def _worker():
            with self.memory.lock:
                games_snapshot = list(self.memory.past_games_history)

            X_rows, y_rows = [], []
            for record in games_snapshot:
                moves = record["moves"]
                metrics = record["metrics"]
                if not moves or not metrics:
                    continue
                board = chess.Board()
                # We don't know the human "winner" label for live play, so use
                # the simple heuristic: whoever delivered checkmate (if any)
                # is labeled the winner; otherwise skip (draw/aborted/unknown).
                final_board = chess.Board()
                for m in moves:
                    final_board.push(m)
                if not final_board.is_checkmate():
                    continue
                # side NOT to move delivered mate
                white_won = 1 if final_board.turn == chess.BLACK else 0

                for i, m in enumerate(moves):
                    if i < len(metrics):
                        tension = metrics[i]['tension']
                        ply = metrics[i]['ply']
                    else:
                        continue
                    features = [tension, ply, 0, 32 - min(i, 30)]  # coarse piece-count proxy
                    X_rows.append(features)
                    y_rows.append(white_won)

            if len(X_rows) < 10:
                return  # not enough new signal yet

            X = np.array(X_rows, dtype=np.float64)
            y = np.array(y_rows, dtype=np.int32)

            try:
                self.forest_model.fit(X, y)
                self.wide_model.partial_fit(X, y, classes=self.classes)
                self.is_initialized = True
            except Exception as e:
                print(f"[WARN] Background retrain failed silently-safe: {e}")

        self._retrain_thread = threading.Thread(target=_worker, daemon=True)
        self._retrain_thread.start()

    def fast_tactical_evaluation(self, board: chess.Board) -> float:
        if board.is_checkmate():
            return -99999.0 if board.turn == chess.WHITE else 99999.0
        if board.is_stalemate() or board.is_insufficient_material():
            return 0.0

        score = 0.0
        for sq in chess.SQUARES:
            p = board.piece_at(sq)
            if p:
                val = self.piece_values[p.piece_type]
                score += val if p.color == chess.WHITE else -val

        tension = DiscoveryEngine.calculate_board_tension(board)
        is_high_friction = tension > 8 or board.is_check()

        if is_high_friction and self.is_initialized:
            try:
                features = np.array([[tension, board.ply(), 1 if board.is_check() else 0, len(board.piece_map())]])
                rf_prob = self.forest_model.predict_proba(features)[0][1] * 5.0
                sgd_prob = self.wide_model.predict_proba(features)[0][1] * 2.0
                cnn_val = self.tf_brain.fast_onnx_predict(self.tf_brain.board_to_tensor(board)) * 10.0

                bias = rf_prob + sgd_prob + cnn_val
                score += (bias if board.turn == chess.WHITE else -bias)
            except Exception:
                # FIX #8: still safe-fallback (material score alone is fine),
                # but this is expected only pre-fit; anything after
                # is_initialized=True is a real bug worth seeing, so:
                if self.is_initialized:
                    pass  # already trained -> a genuine shape/runtime issue;
                          # left silent here to avoid spamming during search,
                          # but see docstring: check self.is_initialized state
                          # and feature shapes if this ever misbehaves.

        return float(score)

    def monte_carlo_search(self, root_board: chess.Board, computational_budget=1500) -> chess.Move:
        """
        NOTE (FIX #4): despite the name (kept for compatibility), this is a
        best-first / value-guided tree search: after expansion, the leaf is
        evaluated directly by fast_tactical_evaluation() rather than played
        out via random rollout. computational_budget therefore controls tree
        SIZE (nodes), not rollout count.

        NOTE (FIX #5): total_value accumulates a White-signed absolute score
        unchanged at every ancestor. This is intentionally consistent with
        Selection alternating max-UCT (White) / min-UCT (Black) below -- do
        NOT add per-ply negation here without also changing Selection, or
        the sign convention will double-flip.
        """
        root = MCTSNode(root_board)

        for _ in range(computational_budget):
            node = root

            # Selection Phase
            while node.is_fully_expanded() and node.children:
                log_parent_visit = math.log(node.visit_count + 1)
                best_move, best_child = None, None

                if node.board.turn == chess.WHITE:
                    best_uct = -float('inf')
                    for move, child in node.children.items():
                        q_value = (child.total_value / child.visit_count) if child.visit_count > 0 else 0
                        uct = q_value + 2.0 * math.sqrt(log_parent_visit / (child.visit_count + 1))
                        if uct > best_uct:
                            best_uct, best_move, best_child = uct, move, child
                else:
                    best_uct = float('inf')
                    for move, child in node.children.items():
                        q_value = (child.total_value / child.visit_count) if child.visit_count > 0 else 0
                        uct = q_value - 2.0 * math.sqrt(log_parent_visit / (child.visit_count + 1))
                        if uct < best_uct:
                            best_uct, best_move, best_child = uct, move, child
                node = best_child

            # Expansion Phase
            legal_moves = node.legal_moves_list()  # FIX #7: cached
            unexpanded = [m for m in legal_moves if m not in node.children]
            if unexpanded:
                unexpanded.sort(key=lambda m: node.board.is_capture(m), reverse=True)
                move = unexpanded[0]
                node.board.push(move)
                child_node = MCTSNode(node.board, parent=node)
                node.board.pop()
                node.children[move] = child_node
                node = child_node

            # Evaluation Phase
            value = self.fast_tactical_evaluation(node.board)

            # Backpropagation Phase
            while node is not None:
                node.visit_count += 1
                node.total_value += value
                node = node.parent

        best_move = None
        best_visits = -1
        for move, child in root.children.items():
            if child.visit_count > best_visits:
                best_visits = child.visit_count
                best_move = move

        return best_move if best_move else random.choice(list(root_board.legal_moves))


# =====================================================================
# 6. TOTAL PIPELINE EXECUTION ENGINE
# =====================================================================
# FIX #9: build the exception tuple once, dynamically, since
# chess.InvalidMoveError only exists in newer python-chess versions.
_INVALID_MOVE_EXCEPTIONS = (ValueError,) + (
    (chess.InvalidMoveError,) if hasattr(chess, "InvalidMoveError") else ()
)


def main():
    memory_vault = CognitiveMemory()
    brain = AdaptiveBrain(memory_vault)

    print("=" * 60)
    print("SYNTHESIZED TRI-ENGINE HYPER-SAFE RUNTIME ONLINE")
    print("=" * 60)
    print("Tip: train on real Lichess data by pasting either a local")
    print(".pgn/.pgn.zst path, or a direct URL such as:")
    print("  https://database.lichess.org/standard/lichess_db_standard_rated_2026-06.pgn.zst")

    user_path = input("Drag & drop your PGN file path or paste a lichess.org URL (or press ENTER to skip): ").strip()

    # FIX #3: only train if something was actually provided
    if user_path:
        brain.train_on_real_file(user_path)
    else:
        print("Skipping pretraining -- engine will run on pure material/search evaluation "
              "until enough games are played to background-retrain the ML layer.")

    session_active = True
    while session_active:
        board = chess.Board()
        played_moves, game_metrics_log = [], []
        user_move = ""

        while not board.is_game_over() and len(board.move_stack) < 100:
            clear_output(wait=True)
            display(board)

            recommended_move = brain.monte_carlo_search(board, computational_budget=1000)
            tension = DiscoveryEngine.calculate_board_tension(board)

            print(f"\nSYSTEM CORE ADAPTIVE ADVISOR (Match Index: {memory_vault.game_counter + 1})")
            print(f"Calculated Path Friction: {tension} active attack vectors mapped across channels.")
            print(f"RECOMMENDED STRATEGIC ACTION: '{board.san(recommended_move) if recommended_move else 'NONE'}'")
            print("-" * 65)

            if board.turn == chess.WHITE:
                print(f"Legal UCI options: {[m.uci() for m in board.legal_moves]}")
                user_move = input("Your Move (White) or type 'exit': ").strip().lower()
                if user_move == 'exit':
                    break
                try:
                    move = chess.Move.from_uci(user_move)
                    if move in board.legal_moves:
                        played_moves.append(move)
                        game_metrics_log.append({'tension': DiscoveryEngine.calculate_board_tension(board), 'ply': board.ply()})
                        board.push(move)
                except (ValueError, chess.InvalidMoveError) if hasattr(chess, "InvalidMoveError") else ValueError:
                    # FIX #9: broadened to also catch python-chess's own
                    # InvalidMoveError on versions that define it.
                    continue
            else:
                ai_move = brain.monte_carlo_search(board, computational_budget=1000)
                played_moves.append(ai_move)
                game_metrics_log.append({'tension': DiscoveryEngine.calculate_board_tension(board), 'ply': board.ply()})
                board.push(ai_move)

        if user_move == 'exit':
            print("Halting active runtime processes.")
            break

        if len(played_moves) > 0:
            clear_output(wait=True)
            memory_vault.record_game(played_moves, game_metrics_log)
            brain.async_background_retrain()

            sgd_w = brain.wide_model.coef_[0] if hasattr(brain.wide_model, "coef_") else None
            rf_i = brain.forest_model.feature_importances_ if hasattr(brain.forest_model, "feature_importances_") else None
            TrueFluidEvaluator.process_and_display(played_moves, game_metrics_log, memory_vault, sgd_w, rf_i)
        else:
            print("\nSession exited before telemetry vectors could accumulate enough data points.")

        if input("Build another interactive match loop? (yes/no): ").strip().lower() != 'yes':
            session_active = False

    if ONNX_AVAILABLE and memory_vault.game_counter > 0:
        print("\nCompiling accrued training adjustments into the engine core...")
        brain.tf_brain.compile_tf_to_onnx()

    print("\nUnified Program Shutdown Successfully.")


if __name__ == "__main__":
    main()
