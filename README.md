# chess_teacher_finder

### ​This project is an interactive, hybrid AI chess engine designed for adaptive gameplay, evaluation, and telemetry synthesis. Built in Python, it integrates traditional move generation with machine learning and value-guided tree search.  
​
### Key Capabilities
​- Tri-Engine AI Core: Combines a Convolutional Neural Network (TensorFlow/ONNX Runtime) for spatial board evaluation, alongside Stochastic Gradient Descent (SGD) and Random Forest classifiers to predict tactical biases.

- ​Value-Guided Tree Search: Uses a modified Monte Carlo Tree Search (MCTS) algorithm to evaluate board control, attack friction, and tension vectors.  
​
- Stockfish Oracle & PGN Integration: Connects with Stockfish or Lichess PGN datasets to automatically train and benchmark engine performance.  
​
- Live Telemetry Matrix: Queries live web APIs (Wikipedia) to generate real-time analytical summaries based on game state metrics.
