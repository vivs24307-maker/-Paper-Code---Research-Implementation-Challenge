import torch

torch.manual_seed(42)

X, Y = [], []
for i in range(64):
    bits = [float(x) for x in format(i, '06b')]
    is_sym = (bits[0] == bits[5]) and (bits[1] == bits[4]) and (bits[2] == bits[3])
    X.append(bits)
    Y.append([1.0 if is_sym else 0.0])

X = torch.tensor(X)
Y = torch.tensor(Y)

w1 = torch.empty(2, 6).uniform_(-0.3, 0.3)
b1 = torch.empty(2, 1).uniform_(-0.3, 0.3)
w2 = torch.empty(1, 2).uniform_(-0.3, 0.3)
b2 = torch.empty(1, 1).uniform_(-0.3, 0.3)

v_w1, v_b1 = torch.zeros_like(w1), torch.zeros_like(b1)
v_w2, v_b2 = torch.zeros_like(w2), torch.zeros_like(b2)

lr, momentum = 0.08, 0.92

for epoch in range(1, 4001):
    x1 = torch.matmul(w1, X.t()) + b1
    y1 = 1.0 / (1.0 + torch.exp(-x1))
    
    x2 = torch.matmul(w2, y1) + b2
    y2 = 1.0 / (1.0 + torch.exp(-x2))
    
    pred = y2.t()
    loss = 0.5 * torch.sum((pred - Y) ** 2).item()
    
    dy2 = y2 - Y.t()
    dx2 = dy2 * (y2 * (1.0 - y2))
    dw2 = torch.matmul(dx2, y1.t())
    db2 = torch.sum(dx2, dim=1, keepdim=True)
    
    dy1 = torch.matmul(w2.t(), dx2)
    dx1 = dy1 * (y1 * (1.0 - y1))
    dw1 = torch.matmul(dx1, X)
    db1 = torch.sum(dx1, dim=1, keepdim=True)
    
    v_w1 = -lr * dw1 + momentum * v_w1
    w1 += v_w1
    v_b1 = -lr * db1 + momentum * v_b1
    b1 += v_b1
    
    v_w2 = -lr * dw2 + momentum * v_w2
    w2 += v_w2
    v_b2 = -lr * db2 + momentum * v_b2
    b2 += v_b2
    
    if epoch % 500 == 0 or epoch == 1:
        acc = ((pred >= 0.5).float() == Y).float().mean().item() * 100
        print(f"Epoch {epoch:4d} | Loss: {loss:11.6f} | Acc: {acc:6.2f}%", flush=True)

print("\nFinal Weights Analysis:", flush=True)
w1_np = w1.numpy()
for i in range(2):
    print(f"Hidden Unit {i+1} Weights: {w1_np[i]}", flush=True)
    print(f"  Check: 0-5 ({w1_np[i,0]:.4f} vs {w1_np[i,5]:.4f}), 1-4 ({w1_np[i,1]:.4f} vs {w1_np[i,4]:.4f}), 2-3 ({w1_np[i,2]:.4f} vs {w1_np[i,3]:.4f})", flush=True)
