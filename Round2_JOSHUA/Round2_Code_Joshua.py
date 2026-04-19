#The provided implementation demonstrates the forward pass. The backward pass is described conceptually using recomputation to avoid storing intermediate activations.
#This is very similar in implementation to FlashAttention 2.0

import numpy as np

def softmax_block(scores):
    max_val = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - max_val)
    sum_exp = np.sum(exp_scores, axis=-1, keepdims=True)
    return exp_scores, sum_exp, max_val

def memory_efficient_attention(Q, K, V, block_size=32):
    n, d = Q.shape

    output = np.zeros((n, d))

    for i in range(0, n, block_size):
        Q_block = Q[i:i+block_size]

        m_i = np.full((Q_block.shape[0], 1), -np.inf)
        l_i = np.zeros((Q_block.shape[0], 1))
        y_i = np.zeros((Q_block.shape[0], d))

        for j in range(0, n, block_size):
            K_block = K[j:j+block_size]
            V_block = V[j:j+block_size]

            scores = Q_block @ K_block.T

            exp_scores, sum_exp, max_block = softmax_block(scores)

            m_new = np.maximum(m_i, max_block)

            l_i = l_i * np.exp(m_i - m_new) + np.sum(
                np.exp(scores - m_new), axis=1, keepdims=True
            )

            y_i = y_i * np.exp(m_i - m_new) + (
                np.exp(scores - m_new) @ V_block
            )

            m_i = m_new

        output[i:i+block_size] = y_i / l_i

    return output



# Example run (n=128, d=32)
if __name__ == "__main__":
    
    n, d = 128, 32
    X = np.random.randn(n, d)

    Q = X
    K = X
    V = X

    out = memory_efficient_attention(Q, K, V, block_size=32)

    print("Output shape:", out.shape)
    print("Sample output:", out[0][:5])

def naive_attention(Q, K, V):
    scores = Q @ K.T
    weights = np.exp(scores - np.max(scores, axis=1, keepdims=True))
    weights = weights / np.sum(weights, axis=1, keepdims=True)
    return weights @ V

if __name__ == "__main__":
    n, d = 128, 32
    X = np.random.randn(n, d)

    Q = X
    K = X
    V = X

    out_fast = memory_efficient_attention(Q, K, V, block_size=32)
    out_naive = naive_attention(Q, K, V)
    error = np.max(np.abs(out_fast - out_naive))
    print("Max difference from naive:", error)