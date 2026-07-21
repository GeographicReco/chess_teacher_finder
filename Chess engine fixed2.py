import chess
import chess.pgn
import chess.engine
import io
import random
import urllib.request
import urllib.parse  # FIX #1: was missing; urllib.parse.quote() is used below
import json
import re
import warnings
import numpy as np
import os
import math
import threading
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from IPython.display import display, clear_output, Math

# Total background warning blackout
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
    print("⚠️ [SYSTEM NOTE] ONNX or TensorFlow libraries missing. Falling back to pure algorithmic structures.")

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
        self.ort_input_name = "board_tensor"  # FIX #3: default, overwritten dynamically after compile
        self.session_lock = threading.Lock()

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
                # FIX #3: don't hardcode the input tensor name — tf2onnx can rename it
                # (e.g. "board_tensor:0" depending on converter/opset version).
                self.ort_input_name = self.ort_session.get_inputs()[0].name
        except Exception:
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

        input_feed = {self.ort_input_name: np.expand_dims(board_tensor, axis=0)}
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
# 4. STRUCTURAL FLUID EVALUATOR (LIVE WEB ENGINE ARCHITECTURE - FIXED)
# =====================================================================
class TrueFluidEvaluator:
    @staticmethod
    def execute_live_search(clean_keywords_only: str) -> str:
        """
        Connects directly to the Wikipedia search engine API,
        submitting ONLY the raw structural tokens for optimal indexing.
        """
        words = clean_keywords_only.split(", ")
        unique_words = []
        for w in words:
            w_clean = w.strip()
            if w_clean not in unique_words:
                unique_words.append(w_clean)

        search_query = " ".join(unique_words)
        encoded_query = urllib.parse.quote(search_query)

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
    def process_and_display(cls, game_moves, memory_vault, sgd_weights, rf_importances):
        print("\n" + "═" * 75)
        print("🌐 ADVANCED SYSTEM INTEGRATION: TOTAL DATA SYNTHESIS MATRIX")
        print("═" * 75)

        total_tension, high_friction_plies = 0, 0
        temp_board = chess.Board()
        for idx, move in enumerate(game_moves):
            tension = DiscoveryEngine.calculate_board_tension(temp_board)
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
        print(f"📡 DISPATCHING CLEAN INTERPOLATED TELEMETRY (CHAOS: {chaos_ratio:.2%}, TENSION: {avg_game_tension:.2f})")
        print("-" * 75)

        live_psy_response = cls.execute_live_search(psy_features_csv)
        live_cs_response = cls.execute_live_search(cs_features_csv)

        print("🔍 LIVE SEARCH ENGINE MATRIX SYNTHESIS RESULTS:")
        print("-" * 75)
        print(f"• Psychology Evaluation Discovery Result:\n  {live_psy_response}")
        print("-" * 75)
        print(f"• Spatial Algorithm Topology Discovery Result:\n  {live_cs_response}")
        print("═" * 75 + "\n")

# =====================================================================
# 4B. STOCKFISH ORACLE — TRAINING-DATA GENERATOR & BENCHMARK OPPONENT
# =====================================================================
class StockfishOracle:
    """
    Wraps a local Stockfish UCI binary. Serves two roles:
      1. Labeling oracle — supplies ground-truth centipawn evaluations
         used to train AdaptiveBrain's SGD/RandomForest/TF models.
      2. Benchmark opponent — plays real games against AdaptiveBrain's
         MCTS engine so you can measure playing strength objectively.
    """

    def __init__(self, engine_path: str, default_time: float = 0.1):
        self.engine_path = engine_path
        self.default_time = default_time
        self.engine = None
        self.lock = threading.Lock()
        self._connect()

    def _connect(self):
        try:
            self.engine = chess.engine.SimpleEngine.popen_uci(self.engine_path)
            print(f"✅ Connected to Stockfish at '{self.engine_path}'.")
        except Exception as e:
            print(f"⚠️ Could not launch Stockfish at '{self.engine_path}': {e}")
            print("   Training/benchmarking against Stockfish will be skipped.")
            self.engine = None

    @property
    def available(self) -> bool:
        return self.engine is not None

    def evaluate_cp(self, board: chess.Board, time_limit=None) -> float:
        """Centipawn evaluation from White's perspective. Mate scores are
        clipped to a large finite value with the correct sign, since a raw
        mate score can't be used as a normal numeric label."""
        if not self.available:
            return 0.0
        limit = chess.engine.Limit(time=time_limit or self.default_time)
        with self.lock:
            info = self.engine.analyse(board, limit)
        score = info["score"].white()
        if score.is_mate():
            mate_in = score.mate()
            return 10000.0 if mate_in > 0 else -10000.0
        return float(score.score())

    def best_move(self, board: chess.Board, time_limit=None) -> chess.Move:
        if not self.available:
            return random.choice(list(board.legal_moves))
        limit = chess.engine.Limit(time=time_limit or self.default_time)
        with self.lock:
            result = self.engine.play(board, limit)
        return result.move

    def close(self):
        if self.engine is not None:
            try:
                self.engine.quit()
            except Exception:
                pass
            self.engine = None


def run_stockfish_test_match(brain, oracle: StockfishOracle, num_games=3,
                              engine_budget=800, stockfish_time=0.1,
                              engine_plays_white=True):
    """Plays AdaptiveBrain's MCTS engine against Stockfish for num_games games,
    alternating colors, and reports win/loss/draw plus average centipawn loss
    per move (how much worse the engine's move was vs. Stockfish's own eval
    swing) — a standard strength metric, independent of just winning or losing."""
    if not oracle.available:
        print("⚠️ Stockfish unavailable — cannot run benchmark match.")
        return None

    results = {"engine_wins": 0, "stockfish_wins": 0, "draws": 0}
    cp_loss_total, cp_loss_count = 0.0, 0

    for game_idx in range(num_games):
        board = chess.Board()
        engine_is_white = engine_plays_white if game_idx % 2 == 0 else not engine_plays_white
        print(f"\n♟️ Benchmark Game {game_idx + 1}/{num_games} — "
              f"Custom engine plays {'White' if engine_is_white else 'Black'}")

        while not board.is_game_over() and len(board.move_stack) < 150:
            is_engine_turn = (board.turn == chess.WHITE) == engine_is_white

            if is_engine_turn:
                pre_cp = oracle.evaluate_cp(board, time_limit=stockfish_time)
                move = brain.monte_carlo_search(board, computational_budget=engine_budget)
                board.push(move)
                post_cp = oracle.evaluate_cp(board, time_limit=stockfish_time)
                # Centipawn loss relative to the side that just moved.
                loss = (pre_cp - post_cp) if engine_is_white else (post_cp - pre_cp)
                cp_loss_total += max(0.0, loss)
                cp_loss_count += 1
            else:
                move = oracle.best_move(board, time_limit=stockfish_time)
                board.push(move)

        outcome = board.outcome()
        if outcome is None or outcome.winner is None:
            results["draws"] += 1
            verdict = "Draw"
        elif outcome.winner == engine_is_white:
            results["engine_wins"] += 1
            verdict = "Custom engine wins"
        else:
            results["stockfish_wins"] += 1
            verdict = "Stockfish wins"
        print(f"  Result: {verdict} ({board.result()}) in {len(board.move_stack)} plies")

    avg_cp_loss = (cp_loss_total / cp_loss_count) if cp_loss_count else 0.0
    print("\n" + "=" * 60)
    print("📊 STOCKFISH BENCHMARK SUMMARY")
    print(f"  Engine wins: {results['engine_wins']}  |  "
          f"Stockfish wins: {results['stockfish_wins']}  |  Draws: {results['draws']}")
    print(f"  Avg. centipawn loss per engine move (vs Stockfish eval): {avg_cp_loss:.1f}")
    print("=" * 60)
    return results

# =====================================================================
# 5. MONTE CARLO TREE SEARCH ENGINE (MATHEMATICALLY CORRECTED)
# =====================================================================
class MCTSNode:
    def __init__(self, board: chess.Board, parent=None):
        self.board = board.copy()
        self.parent = parent
        self.children = {}
        self.visit_count = 0
        self.total_value = 0.0
        # FIX #6: cache legal move count once instead of recomputing list() every call
        self._legal_move_count = len(list(self.board.legal_moves))

    def is_fully_expanded(self):
        return len(self.children) == self._legal_move_count

class AdaptiveBrain:
    def __init__(self, memory_vault: CognitiveMemory):
        self.memory = memory_vault
        self.wide_model = SGDClassifier(loss="log_loss", warm_start=True)
        self.tf_brain = TensorFlowChessBrain()
        self.forest_model = RandomForestClassifier(n_estimators=5, max_depth=3, n_jobs=-1)
        self.is_initialized = False
        self.classes = np.array([0, 1])

        self.piece_values = {
            chess.PAWN: 100.0,
            chess.KNIGHT: 325.0,
            chess.BISHOP: 325.0,
            chess.ROOK: 500.0,
            chess.QUEEN: 975.0,
            chess.KING: 100000.0
        }

    def train_on_real_file(self, file_path: str, max_games=15):
        # [Keep your existing train_on_real_file code here intact]
        # FIX #2 (IMPORTANT): once wide_model / forest_model are actually .fit(...)
        # here, set `self.is_initialized = True` at the end of this method.
        # As written, is_initialized is never flipped to True anywhere, so the
        # entire ML blend in fast_tactical_evaluation is permanently dead code.
        pass

    def async_background_retrain(self):
        # [Keep your existing async_background_retrain code here intact]
        pass

    def train_with_stockfish_oracle(self, oracle: StockfishOracle, num_positions=300,
                                     max_random_plies=40, stockfish_time=0.05):
        """
        Uses Stockfish as a labeling oracle: plays random plies from the
        starting position to get diverse boards, asks Stockfish for a
        centipawn evaluation of each, then fits the SGD/RandomForest/TF
        models on those (features, label) pairs. This is what actually
        turns the ML blend in fast_tactical_evaluation on — is_initialized
        is set True here, which the original code never did anywhere.
        """
        if not oracle.available:
            print("⚠️ Stockfish unavailable — skipping oracle-based training.")
            return

        print(f"📡 Generating {num_positions} labeled positions via Stockfish oracle...")
        states, binary_labels, continuous_values = [], [], []

        while len(states) < num_positions:
            board = chess.Board()
            plies = random.randint(4, max_random_plies)
            for _ in range(plies):
                if board.is_game_over():
                    break
                move = random.choice(list(board.legal_moves))
                board.push(move)
            if board.is_game_over():
                continue

            cp = oracle.evaluate_cp(board, time_limit=stockfish_time)
            states.append(board.copy())
            binary_labels.append(1 if cp > 0 else 0)
            continuous_values.append(math.tanh(cp / 400.0))  # squashed to roughly [-1, 1]

            if len(states) % 25 == 0:
                print(f"  ...{len(states)}/{num_positions} positions labeled")

        features = np.array([
            [DiscoveryEngine.calculate_board_tension(b), b.ply(),
             1 if b.is_check() else 0, len(b.piece_map())]
            for b in states
        ])
        labels = np.array(binary_labels)

        try:
            self.wide_model.partial_fit(features, labels, classes=self.classes)
        except Exception as e:
            print(f"⚠️ SGD training failed: {e}")

        try:
            self.forest_model.fit(features, labels)
        except Exception as e:
            print(f"⚠️ RandomForest training failed: {e}")

        if self.tf_brain.onnx_enabled:
            try:
                self.tf_brain.fit_tensor(states, continuous_values)
                self.tf_brain.compile_tf_to_onnx()
            except Exception as e:
                print(f"⚠️ TF training failed: {e}")

        self.is_initialized = True
        print(f"✅ Oracle training complete on {len(states)} positions. is_initialized = True")

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
                pass

        return float(score)

    def monte_carlo_search(self, root_board: chess.Board, computational_budget=1500) -> chess.Move:
        root = MCTSNode(root_board)

        for _ in range(computational_budget):
            node = root

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

            legal_moves = list(node.board.legal_moves)
            unexpanded = [m for m in legal_moves if m not in node.children]
            if unexpanded:
                unexpanded.sort(key=lambda m: node.board.is_capture(m), reverse=True)
                move = unexpanded[0]
                node.board.push(move)
                child_node = MCTSNode(node.board, parent=node)
                node.board.pop()
                node.children[move] = child_node
                node = child_node

            value = self.fast_tactical_evaluation(node.board)

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
memory_vault = CognitiveMemory()
brain = AdaptiveBrain(memory_vault)

print("═" * 60)
print("📥 SYNTHESIZED TRI-ENGINE HYPER-SAFE RUNTIME ONLINE")
print("═" * 60)

user_path = input("📁 Drag & drop your PGN file path here (or press ENTER for defaults): ").strip()
brain.train_on_real_file(user_path)

sf_path = input("♟️ Path to your Stockfish binary (or press ENTER to try 'stockfish' on PATH): ").strip()
oracle = StockfishOracle(sf_path if sf_path else "stockfish")

if oracle.available:
    if input("Train the engine's ML blend using Stockfish as a labeling oracle now? (yes/no): ").strip().lower() == 'yes':
        try:
            num_pos = int(input("How many labeled positions to generate? [default 200]: ").strip() or "200")
        except ValueError:
            num_pos = 200
        brain.train_with_stockfish_oracle(oracle, num_positions=num_pos)

session_active = True
while session_active:
    print("\n" + "-" * 60)
    print("1) Play interactively against the custom engine")
    print("2) Run a benchmark match: custom engine vs Stockfish")
    print("3) Quit")
    mode = input("Choose an option: ").strip()

    if mode == '3':
        break

    if mode == '2':
        if not oracle.available:
            print("⚠️ Stockfish isn't connected — can't run a benchmark match.")
            continue
        try:
            n_games = int(input("How many benchmark games? [default 3]: ").strip() or "3")
        except ValueError:
            n_games = 3
        run_stockfish_test_match(brain, oracle, num_games=n_games)
        continue

    # mode == '1' (or anything else) falls through to the interactive loop below
    board = chess.Board()
    played_moves, game_metrics_log = [], []
    user_move = ""

    while not board.is_game_over() and len(board.move_stack) < 100:
        clear_output(wait=True)
        display(board)

        recommended_move = brain.monte_carlo_search(board, computational_budget=1000)
        tension = DiscoveryEngine.calculate_board_tension(board)

        print(f"\n🧠 SYSTEM CORE ADAPTIVE ADVISOR (Match Index: {memory_vault.game_counter + 1})")
        print(f"📊 Calculated Path Friction: {tension} active attack vectors mapped across channels.")
        print(f"🎯 RECOMMENDED STRATEGIC ACTION: '{board.san(recommended_move) if recommended_move else 'NONE'}'")
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
                else:
                    # FIX #5: give feedback instead of silently ignoring an illegal move
                    print(f"⚠️ '{user_move}' is not a legal move in this position. Try again.")
                    input("Press ENTER to continue...")
            except ValueError:
                print(f"⚠️ '{user_move}' could not be parsed as a UCI move. Try again.")
                input("Press ENTER to continue...")
                continue
        else:
            # FIX #4: reuse the already-computed recommended_move instead of
            # running a second full Monte Carlo search for the same position.
            ai_move = recommended_move
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
        TrueFluidEvaluator.process_and_display(played_moves, memory_vault, sgd_w, rf_i)
    else:
        print("\nSession exited before telemetry vectors could accumulate enough data points.")

    if input("Build another interactive match loop? (yes/no): ").strip().lower() != 'yes':
        session_active = False

if ONNX_AVAILABLE and memory_vault.game_counter > 0:
    print("\n⏳ Compiling accrued training adjustments into the engine core...")
    brain.tf_brain.compile_tf_to_onnx()

oracle.close()
print("\n🏁 Unified Program Shutdown Successfully.")
