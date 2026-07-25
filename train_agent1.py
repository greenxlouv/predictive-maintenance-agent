"""
train_agent1.py

Agent 1의 LSTM 모델(best_lstm_2d.pth)을 재학습하는 스크립트.
agent1_predictor.py에는 추론 코드(load_model, predict_with_uncertainty)만 있고
학습 코드가 없어서, 이 파일을 따로 만들었음.

모델 구조는 agent1_predictor.py의 LSTMPredictor를 그대로 import해서 사용
(구조가 어긋나면 나중에 agent1_predictor.load_model()로 못 불러오니, 반드시 같은 클래스를 써야 함).

기본 하이퍼파라미터는 팀 실험 보고서의 "Case 2-D"(최종 채택 모델, Test RMSE 31.10) 그대로:
    hidden=64, layers=2, dropout=0.2, epochs=300, Train Loss 기준 Early Stop patience=30
    loss=HuberLoss, optimizer=Adam(lr=1e-3), scheduler=ReduceLROnPlateau(train_loss 기준)

[중요] 체크포인트 선택/Early Stop 기준은 val_loss가 아니라 Train Loss입니다.
한때 "과적합 위험" 때문에 val_loss 기준으로 바꿔봤지만, X_val이 116개뿐이라
노이즈가 심해서 오히려 학습이 채 끝나기도 전(모델이 거의 상수값만 예측하는
단계)에 조기 종료되는 문제가 생겼습니다. 팀이 Case 1 실험에서 이미 "Val 기준
Early Stop이 이 데이터셋에서 불안정하다"는 걸 확인했었고, Case 2-D의 실제
좋은 결과(RMSE 31.10)도 Train Loss 기준으로 나온 것이었어서, 원래 방식으로
되돌렸습니다. val_loss는 참고용으로 로그에만 계속 출력됩니다.

사용법:
    python train_agent1.py --data-dir "C:\\...\\preprocessed" --out best_lstm_2d_v2_fixed.pth

    # 하이퍼파라미터 바꾸고 싶으면
    python train_agent1.py --data-dir ./preprocessed --epochs 300 --patience 30 --batch-size 32 --lr 1e-3
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from agent1_predictor import LSTMPredictor, device


def load_data(data_dir: Path):
    X_train = np.load(data_dir / "X_train.npy")
    y_train = np.load(data_dir / "y_train.npy")
    X_val = np.load(data_dir / "X_val.npy")
    y_val = np.load(data_dir / "y_val.npy")
    return X_train, y_train, X_val, y_val


def make_loader(X, y, batch_size, shuffle):
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    return DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=shuffle)


def run_epoch(model, loader, criterion, optimizer=None):
    """optimizer가 주어지면 학습 모드(가중치 업데이트), 없으면 평가 모드(그냥 loss만 계산)."""
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss, n_samples = 0.0, 0
    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb)
            loss = criterion(preds, yb)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * xb.size(0)
            n_samples += xb.size(0)

    return total_loss / n_samples


def train(
    data_dir: Path,
    out_path: Path,
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.2,
    epochs: int = 300,
    patience: int = 30,
    batch_size: int = 32,
    lr: float = 1e-3,
    scheduler_factor: float = 0.5,
    scheduler_patience: int = 10,
):
    X_train, y_train, X_val, y_val = load_data(data_dir)
    input_size = X_train.shape[-1]

    print(f"X_train: {X_train.shape}, X_val: {X_val.shape}, input_size={input_size}")
    print(f"y_train 중 RUL=0 개수: {(y_train == 0).sum()} (0이면 전처리 파일을 다시 확인하세요)")
    print("[되돌림] 체크포인트 선택/Early Stop 기준을 다시 Train Loss로 되돌림 - "
          "팀 Case 2-D(Test RMSE 31.10)가 실제로 검증한 방식. val_loss는 참고용으로만 출력.")

    train_loader = make_loader(X_train, y_train, batch_size, shuffle=True)
    val_loader = make_loader(X_val, y_val, batch_size, shuffle=False)

    model = LSTMPredictor(
        input_size=input_size, hidden_size=hidden_size,
        num_layers=num_layers, dropout=dropout,
    ).to(device)

    criterion = nn.HuberLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=scheduler_factor, patience=scheduler_patience,
    )

    best_train_loss = float("inf")
    best_val_loss_at_best = None
    best_epoch = None
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        train_loss = run_epoch(model, train_loader, criterion, optimizer)
        val_loss = run_epoch(model, val_loader, criterion, optimizer=None)
        scheduler.step(train_loss)

        # [되돌림] train_loss 기준으로 체크포인트 선택 + early stop 카운트
        improved = train_loss < best_train_loss
        if improved:
            best_train_loss = train_loss
            best_val_loss_at_best = val_loss
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epoch % 10 == 0 or improved:
            marker = " *" if improved else ""
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"epoch {epoch:4d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} "
                  f"| lr={current_lr:.2e}{marker}")

        if epochs_no_improve >= patience:
            print(f"\n조기 종료: train_loss가 {patience}epoch 동안 개선 안 됨 (epoch {epoch})")
            break

    model.load_state_dict(best_state)
    torch.save(model.state_dict(), out_path)
    print(f"\n최종 모델 저장 완료: {out_path}")
    print(f"  선택된 epoch: {best_epoch} | train_loss={best_train_loss:.4f} | 그때의 val_loss={best_val_loss_at_best:.4f}")

    final_val_loss = run_epoch(model, val_loader, criterion, optimizer=None)
    print(f"최종(best) 모델의 val_loss: {final_val_loss:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True,
                         help="X_train.npy, y_train.npy, X_val.npy, y_val.npy 가 있는 폴더")
    parser.add_argument("--out", type=str, default="best_lstm_2d_v2_fixed.pth",
                         help="새로 학습된 가중치를 저장할 파일명")
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--scheduler-factor", type=float, default=0.5,
                         help="ReduceLROnPlateau factor (train_loss 정체 시 LR을 이 비율만큼 감소)")
    parser.add_argument("--scheduler-patience", type=int, default=10,
                         help="ReduceLROnPlateau patience (train_loss가 이 epoch 동안 정체되면 LR 감소)")
    args = parser.parse_args()

    print(f"device: {device}")
    train(
        data_dir=Path(args.data_dir),
        out_path=Path(args.out),
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        lr=args.lr,
        scheduler_factor=args.scheduler_factor,
        scheduler_patience=args.scheduler_patience,
    )


if __name__ == "__main__":
    main()
