import warnings
import json
import gc
import os
from datetime import datetime
from pathlib import Path

# Chỉ áp dụng cho tiến trình chạy file này: dùng tối đa số lõi CPU hiện có
_CPU_COUNT = os.cpu_count() or 1
os.environ.setdefault("OMP_NUM_THREADS", str(_CPU_COUNT))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(_CPU_COUNT))
os.environ.setdefault("MKL_NUM_THREADS", str(_CPU_COUNT))
os.environ.setdefault("NUMEXPR_NUM_THREADS", str(_CPU_COUNT))
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", str(_CPU_COUNT))
os.environ.setdefault("BLIS_NUM_THREADS", str(_CPU_COUNT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wfdb
import seaborn as sns
from tqdm import tqdm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score


warnings.filterwarnings("ignore")


# ===============================
# CẤU HÌNH PHÂN LOẠI NHỊP TIM
# ===============================
AAMI_MAPPING = {
	"N": "N", "L": "N", "R": "N", "e": "N", "j": "N",
	"A": "S", "a": "S", "J": "S", "S": "S",
	"V": "V", "E": "V",
	"F": "F",
	"/": "Q", "f": "Q", "Q": "Q",
}
NON_BEAT_ANNOTATIONS = ["~", "|", "+", "s", "t", "u", "p", "^", "`", '"', "@", "x"]

LABEL_TO_ID = {"N": 0, "S": 1, "V": 2, "F": 3}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}

FS = 360
WINDOW_LEFT = int(0.2 * FS)
WINDOW_RIGHT = int(0.4 * FS)
WINDOW_SIZE = WINDOW_LEFT + WINDOW_RIGHT

MITBIH_RECORDS = [
	"100", "101", "102", "103", "104", "105", "106", "107", "108", "109",
	"111", "112", "113", "114", "115", "116", "117", "118", "119", "121", "122", "123", "124",
	"200", "201", "202", "203", "205", "207", "208", "209", "210", "212", "213", "214",
	"215", "217", "219", "220", "221", "222", "223", "228", "230", "231", "232", "233", "234",
]

# Chế độ an toàn để tránh bị hệ điều hành kill do thiếu RAM khi dò SARIMA
SARIMA_TUNING_MAX_POINTS = 5000
SARIMA_P_MAX = 3
SARIMA_Q_MAX = 3
SARIMA_S_MIN = 216
SARIMA_S_MAX = 300
SARIMA_S_STEP = 12


def configure_runtime_resources(use_all_cores: bool = True) -> None:
	"""Thiết lập affinity và thread cho riêng tiến trình hiện tại."""
	if not use_all_cores:
		return

	cpu_count = os.cpu_count() or 1
	try:
		# Linux: pin process vào toàn bộ core khả dụng (chỉ áp dụng tiến trình này)
		os.sched_setaffinity(0, set(range(cpu_count)))
		print(f"CPU affinity đã set cho tiến trình hiện tại: {cpu_count} cores")
	except Exception:
		print("Không set được CPU affinity, tiếp tục với cấu hình mặc định của hệ điều hành.")

	print(
		"Thread settings -> "
		f"OMP={os.environ.get('OMP_NUM_THREADS')} | "
		f"OPENBLAS={os.environ.get('OPENBLAS_NUM_THREADS')} | "
		f"MKL={os.environ.get('MKL_NUM_THREADS')}"
	)


def configure_aggressive_priority() -> None:
	"""
	Ưu tiên CPU tối đa cho tiến trình hiện tại.
	- Cần quyền root cho mức ưu tiên realtime.
	- Nếu không có quyền, sẽ fallback về mức có thể dùng.
	"""
	pid = os.getpid()
	print(f"PID hiện tại: {pid}")

	# 1) Cố gắng đặt nice = -20 (ưu tiên cao nhất kiểu timesharing)
	try:
		os.nice(-20)
		print("Đã đặt nice=-20 cho tiến trình hiện tại.")
	except Exception:
		print("Không thể set nice=-20 (thường cần sudo).")

	# 2) Cố gắng chuyển sang realtime scheduling để các tiến trình khác phải nhường
	try:
		max_prio = os.sched_get_priority_max(os.SCHED_FIFO)
		os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(max_prio))
		print(f"Đã bật realtime SCHED_FIFO priority={max_prio}.")
	except Exception:
		print("Không thể bật SCHED_FIFO (thường cần sudo/cap_sys_nice).")


def load_wfdb_signal(
	record_name: str = "104",
	pn_dir: str = "mitdb",
	channel: int = 0,
	start_sec: float = 0,
	duration_sec: float | None = None,
) -> tuple[pd.Series, float]:
	"""
	Đọc tín hiệu ECG từ PhysioNet bằng WFDB.

	Trả về:
	- series: pd.Series (index theo giây)
	- fs: tần số lấy mẫu
	"""
	# Đọc metadata trước để lấy fs
	record_info = wfdb.rdrecord(record_name, pn_dir=pn_dir, sampfrom=0, sampto=1)
	fs = float(record_info.fs)

	if duration_sec is None:
		record = wfdb.rdrecord(record_name, pn_dir=pn_dir)
	else:
		sampfrom = int(start_sec * fs)
		sampto = int((start_sec + duration_sec) * fs)
		record = wfdb.rdrecord(record_name, pn_dir=pn_dir, sampfrom=sampfrom, sampto=sampto)
	signal = record.p_signal[:, channel]
	time_index = np.arange(len(signal)) / fs

	series = pd.Series(signal, index=time_index, name=f"ECG_{record_name}")
	return series, fs


def choose_d_by_adf(series: pd.Series, max_d: int = 3, alpha: float = 0.05) -> tuple[int, list[dict]]:
	"""
	Chọn bậc sai phân d nhỏ nhất sao cho chuỗi dừng theo kiểm định ADF (p-value < alpha).
	"""
	adf_results = [] 
	working = series.copy()

	for d in range(max_d + 1):
		adf_stat, p_value, usedlag, nobs, critical_values, _ = adfuller(working.dropna(), autolag="AIC")
		adf_results.append(
			{
				"d": d,
				"adf_stat": adf_stat,
				"p_value": p_value,
				"usedlag": usedlag,
				"nobs": nobs,
				"critical_values": critical_values,
			}
		)

		if p_value < alpha:
			print(f"Chuỗi dừng ở d={d} với p-value={p_value:.6f} < {alpha}")
			return d, adf_results

		working = working.diff().dropna()

	# Nếu không đạt dừng đến max_d thì trả max_d
	return max_d, adf_results


def show_acf_pacf(series: pd.Series, d: int, lags: int = 60) -> None:
	"""
	Vẽ ACF/PACF trên chuỗi sau sai phân d lần để chọn p, q.
	"""
	diff_series = series.copy()
	for _ in range(d):
		diff_series = diff_series.diff().dropna()

	plt.figure(figsize=(14, 8))

	plt.subplot(2, 1, 1)
	plot_acf(diff_series, lags=lags, ax=plt.gca(), title=f"ACF sau sai phân d={d}")

	plt.subplot(2, 1, 2)
	plot_pacf(
		diff_series,
		lags=lags,
		ax=plt.gca(),
		method="ywm",
		title=f"PACF sau sai phân d={d}",
	)

	plt.tight_layout()
	plt.show()


def fit_sarima_and_report(
	series: pd.Series,
	p: int,
	d: int,
	q: int,
	seasonal_order: tuple[int, int, int, int],
):
	"""Fit SARIMA(p,d,q)x(P,D,Q,s) và in tóm tắt."""
	model = SARIMAX(
		series,
		order=(p, d, q),
		seasonal_order=seasonal_order,
		enforce_stationarity=False,
		enforce_invertibility=False,
	)
	result = model.fit(disp=False)
	print("\n================ SARIMA RESULT ================")
	print(f"Order: (p,d,q)=({p},{d},{q}) | seasonal_order={seasonal_order}")
	print(f"BIC={result.bic:.3f} | AIC={result.aic:.3f}")
	return result


def search_best_sarima_bic(
	series: pd.Series,
	d: int,
	p_max: int = SARIMA_P_MAX,
	q_max: int = SARIMA_Q_MAX,
	seasonal_order_fixed: tuple[int, int, int] = (1, 0, 1),
	s_min: int = SARIMA_S_MIN,
	s_max: int = SARIMA_S_MAX,
	s_step: int = SARIMA_S_STEP,
	max_points_for_tuning: int = SARIMA_TUNING_MAX_POINTS,
) -> pd.DataFrame:
	"""Dò p,q và cả s theo BIC cho SARIMA; giữ cố định P,D,Q."""
	rows = []
	work = series.dropna()
	if len(work) > max_points_for_tuning:
		work = work.iloc[:max_points_for_tuning]
	print(f"SARIMA tuning dùng {len(work)} điểm dữ liệu | p<= {p_max}, q<= {q_max}, s={s_min}..{s_max} (step={s_step})")

	P, D, Q = seasonal_order_fixed
	for s in range(s_min, s_max + 1, s_step):
		for p in range(p_max + 1):
			for q in range(q_max + 1):
				try:
					result = SARIMAX(
						work,
						order=(p, d, q),
						seasonal_order=(P, D, Q, s),
						enforce_stationarity=False,
						enforce_invertibility=False,
					).fit(disp=False)
					rows.append({"p": p, "d": d, "q": q, "s": s, "bic": result.bic, "aic": result.aic})
					print(f"Đã fit SARIMA(p={p},d={d},q={q})x(P={P},D={D},Q={Q},s={s}) -> BIC={result.bic:.3f}")
					del result
					gc.collect()
				except MemoryError:
					print("Thiếu RAM khi fit SARIMA, dừng dò thêm tham số.")
					break
				except Exception:
					continue

	if not rows:
		return pd.DataFrame(columns=["p", "d", "q", "s", "bic", "aic"])

	return pd.DataFrame(rows).sort_values("bic").reset_index(drop=True)


def plot_fit_and_forecast(series: pd.Series, result, forecast_steps: int = 360) -> None:
	"""Vẽ chuỗi gốc + fitted + forecast từ mô hình SARIMA đã fit."""
	fitted = result.fittedvalues
	forecast = result.forecast(steps=forecast_steps)

	plt.figure(figsize=(14, 5))
	plt.plot(series.index, series.values, label="Dữ liệu gốc", linewidth=1)

	fitted_index = series.index[-len(fitted):]
	plt.plot(fitted_index, fitted.values, label="Fitted", linewidth=1.2)

	dt = series.index[1] - series.index[0]
	forecast_index = np.arange(series.index[-1] + dt, series.index[-1] + (forecast_steps + 1) * dt, dt)
	plt.plot(forecast_index, forecast.values, label=f"Forecast {forecast_steps} bước", linewidth=1.2)

	plt.title("SARIMA - Fitted và Forecast")
	plt.xlabel("Thời gian (giây)")
	plt.ylabel("Biên độ ECG")
	plt.grid(alpha=0.3)
	plt.legend()
	plt.tight_layout()
	plt.show()


def extract_heartbeats(record_names: list[str], pn_dir: str = "mitdb", max_records: int = 8) -> tuple[np.ndarray, np.ndarray]:
	"""
	Cắt từng nhịp tim theo annotation R-peak từ MIT-BIH.
	"""
	X, y = [], []
	print(f"Bắt đầu trích xuất dữ liệu từ {min(len(record_names), max_records)} bản ghi...")

	for record_name in tqdm(record_names[:max_records]):
		try:
			record = wfdb.rdrecord(record_name, pn_dir=pn_dir)
			ann = wfdb.rdann(record_name, "atr", pn_dir=pn_dir)
			signal = record.p_signal[:, 0]

			for symbol, sample_idx in zip(ann.symbol, ann.sample):
				if symbol in NON_BEAT_ANNOTATIONS:
					continue

				start_idx = sample_idx - WINDOW_LEFT
				end_idx = sample_idx + WINDOW_RIGHT
				if start_idx < 0 or end_idx >= len(signal):
					continue

				label = AAMI_MAPPING.get(symbol, "Q")

				# Bỏ nhãn Unknown (Q) theo yêu cầu
				if label == "Q":
					continue

				X.append(signal[start_idx:end_idx])
				y.append(LABEL_TO_ID[label])
		except Exception as e:
			print(f"Lỗi bản ghi {record_name}: {e}")

	return np.array(X), np.array(y)


def zscore_per_beat(X: np.ndarray) -> np.ndarray:
	"""Chuẩn hóa từng nhịp tim về mean=0, std=1."""
	mu = X.mean(axis=1, keepdims=True)
	std = X.std(axis=1, keepdims=True)
	std = np.where(std < 1e-8, 1.0, std)
	return (X - mu) / std


def build_feature_matrix_with_pdq(X_beats: np.ndarray, p: int = 4, d: int = 0, q: int = 4) -> np.ndarray:
	"""
	Ghép đặc trưng cửa sổ nhịp tim + 3 đặc trưng [p, d, q].
	"""
	X_norm = zscore_per_beat(X_beats)
	pdq_feat = np.tile(np.array([[p, d, q]], dtype=np.float32), (len(X_norm), 1))
	return np.hstack([X_norm, pdq_feat])


def save_sarima_arima_params(
	P: int,
	D: int,
	Q: int,
	s: int,
	p: int,
	d: int,
	q: int,
	acc: float | None = None,
	macro_f1: float | None = None,
	weighted_f1: float | None = None,
	output_file: str = "sarima_arima_params.json",
) -> Path:
	"""Lưu tham số và metric ra file JSON."""
	payload = {
		"saved_at": datetime.now().isoformat(timespec="seconds"),
		"seasonal_params": {"P": P, "D": D, "Q": Q, "s": s},
		"non_seasonal_params": {"p": p, "d": d, "q": q},
		"metrics": {
			"accuracy": acc,
			"macro_f1": macro_f1,
			"weighted_f1": weighted_f1,
		},
	}

	out_path = Path(__file__).resolve().parent / output_file
	out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
	print(f"Đã lưu tham số vào: {out_path}")
	return out_path


def save_sarima_pdq_bic_to_excel(
	table: pd.DataFrame,
	output_file: str = "sarima_pdq_bic.xlsx",
) -> Path:
	"""Lưu bảng kết quả dò SARIMA (p,d,q,s,bic,aic) ra file Excel."""
	if table.empty:
		raise ValueError("Bảng kết quả SARIMA rỗng, không thể xuất Excel.")

	cols_priority = ["p", "d", "q", "s", "bic", "aic"]
	cols = [c for c in cols_priority if c in table.columns]
	export_df = table[cols].copy()

	out_path = Path(__file__).resolve().parent / output_file
	export_df.to_excel(out_path, index=False, sheet_name="sarima_bic")
	print(f"Đã lưu bảng p,d,q và BIC vào: {out_path}")
	return out_path


def run_logistic_regression_with_arima_features(
	record_names: list[str] | None = None,
	max_records: int | None = None,
	max_samples: int | None = 2500,
	p: int = 4,
	d: int = 0,
	q: int = 4,
) -> tuple[float, float, float]:
	"""
	Task classification với Logistic Regression cơ bản:
	- Cắt window từng nhịp
	- Thêm 3 đặc trưng [p, d, q] lấy từ bước SARIMA tuning
	- Dự đoán nhãn AAMI
	"""
	print(f"\n================ CLASSIFICATION: Logistic + [p,d,q] = [{p},{d},{q}] ================")
	if record_names is None:
		record_names = MITBIH_RECORDS
	if max_records is None:
		max_records = len(record_names)

	print(f"Dùng record: {record_names[:max_records]} | lấy toàn bộ độ dài chuỗi mỗi record")
	X, y = extract_heartbeats(record_names, pn_dir="mitdb", max_records=max_records)

	if len(X) == 0:
		print("Không có dữ liệu. Kiểm tra mạng hoặc PhysioNet.")
		return 0.0, 0.0, 0.0

	if max_samples is not None and len(X) > max_samples:
		rng = np.random.default_rng(42)
		idx = rng.choice(len(X), size=max_samples, replace=False)
		X = X[idx]
		y = y[idx]

	print(f"Số mẫu dùng train/test: {len(X)}")
	classes, counts = np.unique(y, return_counts=True)
	for c, n in zip(classes, counts):
		print(f"- Lớp {ID_TO_LABEL[c]}: {n} mẫu")

	X_features = build_feature_matrix_with_pdq(X, p=p, d=d, q=q)
	print(f"Kích thước đặc trưng: {X_features.shape} (216 điểm nhịp + 3 đặc trưng p,d,q)")

	# Nếu lớp quá hiếm (ít hơn 2 mẫu), không thể stratify
	class_counts = np.bincount(y)
	can_stratify = np.all(class_counts[class_counts > 0] >= 2)

	X_train, X_test, y_train, y_test = train_test_split(
		X_features,
		y,
		test_size=0.2,
		stratify=y if can_stratify else None,
		random_state=42,
	)

	scaler = StandardScaler()
	X_train_sc = scaler.fit_transform(X_train)
	X_test_sc = scaler.transform(X_test)

	clf = LogisticRegression(
		max_iter=5000,
		multi_class="multinomial",
		solver="lbfgs",
		class_weight="balanced",
		n_jobs=None,
	)
	clf.fit(X_train_sc, y_train)
	y_pred = clf.predict(X_test_sc)
	acc = (y_pred == y_test).mean()
	macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
	weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

	labels_present = np.unique(y_test)
	target_names = [ID_TO_LABEL[i] for i in labels_present]

	print("\n--- Classification report ---")
	print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))
	print(f"Accuracy test: {acc:.4f}")
	print(f"Macro-F1: {macro_f1:.4f}")
	print(f"Weighted-F1: {weighted_f1:.4f}")

	print("\n--- 30 dự đoán đầu tiên (true -> pred) ---")
	for i in range(min(30, len(y_test))):
		true_label = ID_TO_LABEL[int(y_test[i])]
		pred_label = ID_TO_LABEL[int(y_pred[i])]
		print(f"{i+1:02d}. {true_label} -> {pred_label}")

	cm = confusion_matrix(y_test, y_pred, labels=labels_present)
	plt.figure(figsize=(8, 6))
	sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=target_names, yticklabels=target_names)
	plt.title("Confusion Matrix - Logistic Regression + [p,d,q]")
	plt.xlabel("Predicted")
	plt.ylabel("True")
	plt.tight_layout()
	plt.show()

	return acc, macro_f1, weighted_f1


def main() -> None:
	configure_runtime_resources(use_all_cores=True)
	configure_aggressive_priority()

	# ========== STEP 1: CHỌN d TỪ MỘT RECORD DUY NHẤT ==========
	adf_record = "104"
	pn_dir = "mitdb"
	series, fs = load_wfdb_signal(record_name=adf_record, pn_dir=pn_dir, duration_sec=None)
	print(f"Record tham chiếu: {adf_record} | fs={fs}Hz | số mẫu={len(series)}")

	# ========== STEP 2: DÒ p,q CHO SARIMA THEO BIC ==========
	seasonal_order_fixed = (1, 0, 1)
	table = search_best_sarima_bic(
		series=series,
		d=0,
		p_max=SARIMA_P_MAX,
		q_max=SARIMA_Q_MAX,
		seasonal_order_fixed=seasonal_order_fixed,
		s_min=SARIMA_S_MIN,
		s_max=SARIMA_S_MAX,
		s_step=SARIMA_S_STEP,
		max_points_for_tuning=SARIMA_TUNING_MAX_POINTS,
	)
	if table.empty:
		print("Không dò được SARIMA hợp lệ.")
		return

	print("\n===== TOP 10 SARIMA TỐT NHẤT THEO BIC =====")
	print(table.head(10).to_string(index=False))
	save_sarima_pdq_bic_to_excel(table, output_file="sarima_pdq_bic.xlsx")

	best = table.iloc[0]
	p_best = int(best["p"])
	q_best = int(best["q"])
	s_best = int(best["s"])
	P_fixed, D_fixed, Q_fixed = seasonal_order_fixed
	print(f"\n=> Chọn theo BIC: p={p_best}, d={0}, q={q_best}, s={s_best}")

	# ========== STEP 3: CLASSIFICATION TRÊN TẤT CẢ RECORD ==========
	print("\n===== CLASSIFICATION TRÊN TOÀN BỘ RECORD =====")
	acc, macro_f1, weighted_f1 = run_logistic_regression_with_arima_features(
		record_names=MITBIH_RECORDS,
		max_records=len(MITBIH_RECORDS),
		max_samples=8000,
		p=p_best,
		d=0,
		q=q_best,
	)

	print("\n===== TÓM TẮT CUỐI =====")
	print(f"Dùng tham số từ SARIMA: p={p_best}, d={0}, q={q_best}, s={s_best}")
	print(f"Accuracy={acc:.4f} | Macro-F1={macro_f1:.4f} | Weighted-F1={weighted_f1:.4f}")

	save_sarima_arima_params(
		P=P_fixed,
		D=D_fixed,
		Q=Q_fixed,
		s=s_best,
		p=p_best,
		d=0,
		q=q_best,
		acc=acc,
		macro_f1=macro_f1,
		weighted_f1=weighted_f1,
	)


if __name__ == "__main__":
	main()

