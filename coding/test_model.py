from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wfdb
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX


RECORD_NAME = "104"
PN_DIR = "mitdb"
CHANNEL = 0

# Dùng một đoạn đủ lớn để fit nhanh và ổn định
MAX_POINTS = 12000
TEST_STEPS = 720  # 2 giây với fs=360

# Tham số user yêu cầu
P, D, Q, S = 1, 0, 1, 216
p, d, q = 1, 0, 3


def load_ecg(record_name: str, pn_dir: str, channel: int = 0) -> tuple[np.ndarray, float]:
	record = wfdb.rdrecord(record_name, pn_dir=pn_dir)
	fs = float(record.fs)
	signal = record.p_signal[:, channel].astype(np.float64)
	return signal, fs


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
	eps = 1e-8
	err = y_true - y_pred
	mae = float(np.mean(np.abs(err)))
	rmse = float(np.sqrt(np.mean(err**2)))
	mape = float(np.mean(np.abs(err) / (np.abs(y_true) + eps)) * 100.0)
	smape = float(np.mean(2.0 * np.abs(err) / (np.abs(y_true) + np.abs(y_pred) + eps)) * 100.0)
	return {"MAE": mae, "RMSE": rmse, "MAPE_percent": mape, "sMAPE_percent": smape}


def fit_and_forecast(train: np.ndarray, steps: int) -> tuple[np.ndarray, np.ndarray]:
	# SARIMA(1,0,3)x(1,0,1,216)
	sarima = SARIMAX(
		train,
		order=(p, d, q),
		seasonal_order=(P, D, Q, S),
		enforce_stationarity=False,
		enforce_invertibility=False,
	)
	sarima_res = sarima.fit(disp=False)
	sarima_fc = sarima_res.forecast(steps=steps)

	# ARIMA(1,0,3)
	arima = ARIMA(train, order=(p, d, q))
	arima_res = arima.fit()
	arima_fc = arima_res.forecast(steps=steps)

	return np.asarray(sarima_fc), np.asarray(arima_fc)


def main() -> None:
	signal, fs = load_ecg(RECORD_NAME, PN_DIR, CHANNEL)
	series = signal[:MAX_POINTS]

	if len(series) <= TEST_STEPS + 100:
		raise ValueError("Không đủ điểm dữ liệu để tách train/test.")

	train = series[:-TEST_STEPS]
	test = series[-TEST_STEPS:]

	print(f"Record={RECORD_NAME} | fs={fs} | total_used={len(series)} | train={len(train)} | test={len(test)}")
	print(f"SARIMA: ({p},{d},{q})x({P},{D},{Q},{S}) | ARIMA: ({p},{d},{q})")

	sarima_fc, arima_fc = fit_and_forecast(train, TEST_STEPS)

	sarima_metrics = regression_metrics(test, sarima_fc)
	arima_metrics = regression_metrics(test, arima_fc)

	print("\n===== METRICS (lower is better) =====")
	print("SARIMA:", sarima_metrics)
	print("ARIMA :", arima_metrics)

	out_dir = Path(__file__).resolve().parent

	# Lưu metrics
	payload = {
		"config": {
			"record": RECORD_NAME,
			"fs": fs,
			"max_points": MAX_POINTS,
			"test_steps": TEST_STEPS,
			"sarima": {"p": p, "d": d, "q": q, "P": P, "D": D, "Q": Q, "s": S},
			"arima": {"p": p, "d": d, "q": q},
		},
		"metrics": {"sarima": sarima_metrics, "arima": arima_metrics},
	}
	metrics_path = out_dir / "forecast_metrics_sarima_vs_arima.json"
	metrics_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

	# Lưu forecast
	idx = np.arange(TEST_STEPS)
	df = pd.DataFrame({
		"step": idx,
		"y_true": test,
		"sarima_forecast": sarima_fc,
		"arima_forecast": arima_fc,
	})
	csv_path = out_dir / "forecast_values_sarima_vs_arima.csv"
	df.to_csv(csv_path, index=False)

	# Plot
	plt.figure(figsize=(12, 5))
	plt.plot(idx, test, label="True", linewidth=1.5)
	plt.plot(idx, sarima_fc, label="SARIMA forecast", linewidth=1.2)
	plt.plot(idx, arima_fc, label="ARIMA forecast", linewidth=1.2)
	plt.title("Forecast Comparison: SARIMA vs ARIMA")
	plt.xlabel("Forecast step")
	plt.ylabel("ECG amplitude")
	plt.grid(alpha=0.3)
	plt.legend()
	plt.tight_layout()
	fig_path = out_dir / "forecast_compare_sarima_vs_arima.png"
	plt.savefig(fig_path, dpi=150)
	plt.show()

	print(f"\nĐã lưu metrics: {metrics_path}")
	print(f"Đã lưu forecast values: {csv_path}")
	print(f"Đã lưu biểu đồ: {fig_path}")


if __name__ == "__main__":
	main()
