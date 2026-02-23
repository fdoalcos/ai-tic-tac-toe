from __future__ import annotations

from flask import Flask, request, jsonify, send_from_directory
import math
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

app = Flask(__name__, static_folder="static")

# -----------------------------
# Game logic
# -----------------------------
HUMAN = "X"
AI = "O"
EMPTY = " "

WIN_LINES = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),
    (0, 3, 6), (1, 4, 7), (2, 5, 8),
    (0, 4, 8), (2, 4, 6)
]

def check_winner(board: List[str]) -> Optional[str]:
    for a, b, c in WIN_LINES:
        if board[a] != EMPTY and board[a] == board[b] == board[c]:
            return board[a]
    if EMPTY not in board:
        return "DRAW"
    return None

def legal_moves(board: List[str]) -> List[int]:
    return [i for i, v in enumerate(board) if v == EMPTY]

def apply_move(board: List[str], idx: int, player: str) -> List[str]:
    nb = board[:]
    nb[idx] = player
    return nb

def other(player: str) -> str:
    return AI if player == HUMAN else HUMAN

def terminal_score(board: List[str]) -> Optional[int]:
    w = check_winner(board)
    if w is None:
        return None
    if w == AI:
        return 10
    if w == HUMAN:
        return -10
    return 0

# -----------------------------
# Heuristic (for depth-limited)
# -----------------------------
def heuristic_eval(board: List[str]) -> int:
    """
    Simple heuristic: favor open lines for AI; penalize open lines for HUMAN.
    """
    score = 0
    for a, b, c in WIN_LINES:
        line = [board[a], board[b], board[c]]
        if HUMAN not in line:
            score += line.count(AI)
        if AI not in line:
            score -= line.count(HUMAN)
    return score

# -----------------------------
# Minimax
# -----------------------------
def minimax(board: List[str], player: str) -> Tuple[int, Optional[int], int]:
    """
    Returns (best_score, best_move, nodes_expanded)
    """
    nodes = 1
    ts = terminal_score(board)
    if ts is not None:
        return ts, None, nodes

    moves = legal_moves(board)
    best_move = None

    if player == AI:
        best_score = -math.inf
        for m in moves:
            s, _, n = minimax(apply_move(board, m, player), other(player))
            nodes += n
            if s > best_score:
                best_score, best_move = s, m
        return int(best_score), best_move, nodes
    else:
        best_score = math.inf
        for m in moves:
            s, _, n = minimax(apply_move(board, m, player), other(player))
            nodes += n
            if s < best_score:
                best_score, best_move = s, m
        return int(best_score), best_move, nodes

def minimax_depth_limited(board: List[str], player: str, depth: int) -> Tuple[int, Optional[int], int]:
    """
    Returns (best_score, best_move, nodes_expanded)
    """
    nodes = 1
    ts = terminal_score(board)
    if ts is not None:
        return ts, None, nodes
    if depth == 0:
        return heuristic_eval(board), None, nodes

    moves = legal_moves(board)
    best_move = None

    if player == AI:
        best_score = -math.inf
        for m in moves:
            s, _, n = minimax_depth_limited(apply_move(board, m, player), other(player), depth - 1)
            nodes += n
            if s > best_score:
                best_score, best_move = s, m
        return int(best_score), best_move, nodes
    else:
        best_score = math.inf
        for m in moves:
            s, _, n = minimax_depth_limited(apply_move(board, m, player), other(player), depth - 1)
            nodes += n
            if s < best_score:
                best_score, best_move = s, m
        return int(best_score), best_move, nodes

def minimax_alpha_beta(board: List[str], player: str, alpha: float, beta: float) -> Tuple[int, Optional[int], int, int]:
    """
    Returns (best_score, best_move, nodes_expanded, prunes)
    """
    nodes = 1
    prunes = 0

    ts = terminal_score(board)
    if ts is not None:
        return ts, None, nodes, prunes

    moves = legal_moves(board)
    best_move = None

    if player == AI:
        best_score = -math.inf
        for m in moves:
            s, _, n, p = minimax_alpha_beta(apply_move(board, m, player), other(player), alpha, beta)
            nodes += n
            prunes += p
            if s > best_score:
                best_score, best_move = s, m
            alpha = max(alpha, best_score)
            if alpha >= beta:
                prunes += 1
                break
        return int(best_score), best_move, nodes, prunes
    else:
        best_score = math.inf
        for m in moves:
            s, _, n, p = minimax_alpha_beta(apply_move(board, m, player), other(player), alpha, beta)
            nodes += n
            prunes += p
            if s < best_score:
                best_score, best_move = s, m
            beta = min(beta, best_score)
            if alpha >= beta:
                prunes += 1
                break
        return int(best_score), best_move, nodes, prunes

# -----------------------------
# MCTS
# -----------------------------
@dataclass
class MCTSNode:
    board: Tuple[str, ...]
    player_to_move: str
    parent: Optional["MCTSNode"]
    move_from_parent: Optional[int]
    children: Dict[int, "MCTSNode"]
    untried_moves: List[int]
    visits: int
    value: float  # total value from AI perspective

    def is_terminal(self) -> bool:
        return check_winner(list(self.board)) is not None

def mcts_select_child(node: MCTSNode, c: float = 1.4) -> MCTSNode:
    best_score = -math.inf
    best_child = None
    for _, child in node.children.items():
        if child.visits == 0:
            ucb = math.inf
        else:
            exploitation = child.value / child.visits
            exploration = c * math.sqrt(math.log(max(1, node.visits)) / child.visits)
            ucb = exploitation + exploration
        if ucb > best_score:
            best_score = ucb
            best_child = child
    return best_child  # type: ignore

def mcts_expand(node: MCTSNode) -> MCTSNode:
    m = node.untried_moves.pop()
    nb = list(node.board)
    nb[m] = node.player_to_move
    child = MCTSNode(
        board=tuple(nb),
        player_to_move=other(node.player_to_move),
        parent=node,
        move_from_parent=m,
        children={},
        untried_moves=legal_moves(nb),
        visits=0,
        value=0.0,
    )
    node.children[m] = child
    return child

def mcts_rollout(board: List[str], player_to_move: str) -> float:
    b = board[:]
    p = player_to_move
    w = check_winner(b)
    while w is None:
        m = random.choice(legal_moves(b))
        b[m] = p
        p = other(p)
        w = check_winner(b)

    if w == AI:
        return 1.0
    if w == HUMAN:
        return -1.0
    return 0.0

def mcts_backpropagate(node: MCTSNode, reward: float):
    cur = node
    while cur is not None:
        cur.visits += 1
        cur.value += reward
        cur = cur.parent

def mcts_best_move(board: List[str], player_to_move: str, rollouts: int) -> Tuple[Optional[int], int]:
    root = MCTSNode(
        board=tuple(board),
        player_to_move=player_to_move,
        parent=None,
        move_from_parent=None,
        children={},
        untried_moves=legal_moves(board),
        visits=0,
        value=0.0,
    )
    if root.is_terminal():
        return None, 0

    for _ in range(max(1, rollouts)):
        node = root

        # Selection
        while (not node.is_terminal()) and (not node.untried_moves) and node.children:
            node = mcts_select_child(node)

        # Expansion
        if (not node.is_terminal()) and node.untried_moves:
            node = mcts_expand(node)

        # Simulation
        reward = mcts_rollout(list(node.board), node.player_to_move)

        # Backprop
        mcts_backpropagate(node, reward)

    if not root.children:
        return None, root.visits

    best = max(root.children.values(), key=lambda c: c.visits)
    return best.move_from_parent, root.visits

# -----------------------------
# Server-side game state
# -----------------------------
STATE = {
    "board": [EMPTY] * 9,
    "player_score": 0,
    "ai_score": 0,
}

def state_payload(extra: Optional[dict] = None):
    payload = {
        "board": STATE["board"],
        "winner": check_winner(STATE["board"]),
        "player_score": STATE["player_score"],
        "ai_score": STATE["ai_score"],
    }
    if extra:
        payload.update(extra)
    return payload

def ai_choose_move(algorithm: str, depth: int, rollouts: int):
    board = STATE["board"]
    if check_winner(board) is not None:
        return None, {"nodes": 0, "prunes": 0, "ms": 0}

    if algorithm == "minimax":
        score, move, nodes = minimax(board, AI)
        return move, {"nodes": nodes, "score": score}
    if algorithm == "depth-minimax":
        score, move, nodes = minimax_depth_limited(board, AI, depth)
        return move, {"nodes": nodes, "score": score, "depth": depth}
    if algorithm == "alpha-beta":
        score, move, nodes, prunes = minimax_alpha_beta(board, AI, -math.inf, math.inf)
        return move, {"nodes": nodes, "prunes": prunes, "score": score}
    if algorithm == "mcts":
        move, visits = mcts_best_move(board, AI, rollouts)
        return move, {"nodes": visits, "rollouts": rollouts}

    # default
    score, move, nodes = minimax(board, AI)
    return move, {"nodes": nodes, "score": score}

# -----------------------------
# Routes
# -----------------------------
@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")

@app.post("/api/reset")
def api_reset():
    STATE["board"] = [EMPTY] * 9
    return jsonify(state_payload({"log": ["> System reset. New game started."]}))

@app.post("/api/human_move")
def api_human_move():
    data = request.get_json(force=True)
    idx = int(data.get("index", -1))
    if idx < 0 or idx > 8:
        return jsonify(state_payload({"error": "Invalid index"})), 400
    if STATE["board"][idx] != EMPTY:
        return jsonify(state_payload({"error": "Cell not empty"})), 400
    if check_winner(STATE["board"]) is not None:
        return jsonify(state_payload({"error": "Game already ended"})), 400

    STATE["board"][idx] = HUMAN
    w = check_winner(STATE["board"])
    logs = [f"> Player moved to index {idx}."]

    if w == HUMAN:
        STATE["player_score"] += 1
        logs.append("> Player wins! +1 Player Score.")
    elif w == "DRAW":
        logs.append("> Game is a draw.")

    return jsonify(state_payload({"log": logs}))

@app.post("/api/ai_move")
def api_ai_move():
    data = request.get_json(force=True)
    algorithm = data.get("algorithm", "minimax")  # minimax|depth-minimax|alpha-beta|mcts
    depth = int(data.get("depth", 4))
    rollouts = int(data.get("rollouts", 1000))

    if check_winner(STATE["board"]) is not None:
        return jsonify(state_payload({"log": ["> No AI move: game already ended."]}))

    move, stats = ai_choose_move(algorithm, depth, rollouts)
    logs = [f"> AI thinking ({algorithm})..."]

    if move is None:
        logs.append("> No legal AI move.")
        return jsonify(state_payload({"log": logs, "stats": stats}))

    STATE["board"][move] = AI
    logs.append(f"> AI placed O at index {move}.")
    logs.append(f"> NODES: {stats.get('nodes', 0)}" + (f" | PRUNES: {stats.get('prunes', 0)}" if "prunes" in stats else ""))

    w = check_winner(STATE["board"])
    if w == AI:
        STATE["ai_score"] += 1
        logs.append("> AI wins! +1 AI Score.")
    elif w == "DRAW":
        logs.append("> Game is a draw.")

    return jsonify(state_payload({"log": logs, "stats": stats}))

if __name__ == "__main__":
    # Go to http://127.0.0.1:5000
    app.run(debug=True)