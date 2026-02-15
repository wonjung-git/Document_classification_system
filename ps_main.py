import sys
import re
import os
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QFrame, QStackedWidget,
                             QFileDialog, QMessageBox, QRadioButton, 
                             QTableWidget, QTableWidgetItem, 
                             QHeaderView, QComboBox, QDialog, QProgressDialog, QSpinBox, QLineEdit, QAbstractItemView)
from PySide6.QtCore import Qt, Signal, QThread, QEvent, QSize
from PySide6.QtGui import QCursor, QFont, QColor, QIcon, QPixmap, QKeySequence, QPainter
import ctypes
import string
from tokenization_kobert import KoBertTokenizer


# --- 이미지 경로 설정 ---
def resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, relative_path)

# --- 이미지 화질 좋아지는 함수 ---
def hd_pixmap(ref_widget, path: str, logical_px: int) -> QPixmap:
    dpr = ref_widget.devicePixelRatioF()
    pm = QPixmap(path)
    if pm.isNull():
        return QPixmap()

    pm = pm.scaled(int(logical_px * dpr), int(logical_px * dpr),
                   Qt.KeepAspectRatio, Qt.SmoothTransformation)
    pm.setDevicePixelRatio(dpr)
    return pm

ML_ICON_PATH = resource_path("assets/ml_icon.png") #1p
SEARCH_ICON_PATH = resource_path("assets/search_icon.png") #1p
TITLE_ICON_PATH = resource_path("assets/title_icon.png") #2p
DOC_ICON_PATH = resource_path("assets/doc_icon.png") #2p
DASH_ICON_PATH = resource_path("assets/dash_icon.png") #2p


# --- 직위별 가중치 설정 ---
POSITION_WEIGHTS = {
    '시장': (2.5, 1.8, 0.8, 0.2), '군수': (2.5, 1.8, 0.8, 0.2), '구청장': (2.5, 1.8, 0.8, 0.2),
    '부시장': (2.3, 2.0, 0.8, 0.2), '부군수': (2.3, 2.0, 0.8, 0.2), '부구청장': (2.3, 2.0, 0.8, 0.2),
    '실장': (2.0, 1.5, 1.0, 0.5), '단장': (2.0, 1.5, 1.0, 0.5),
    '국장': (1.2, 1.2, 1.2, 0.8), '서기관': (1.2, 1.2, 1.2, 0.8),
    '과장': (1.1, 1.2, 1.3, 1.2), '동장': (1.1, 1.2, 1.3, 1.2),
    '팀장': (1.0, 1.1, 1.2, 1.4), '담당': (1.0, 1.1, 1.2, 1.4),
    '주무관': (0.8, 0.8, 1.0, 1.5), '수습': (0.8, 0.8, 0.8, 1.5)
}


# --- 등급 조정 방식 ---
class QuotaAllocator:
    def __init__(self, df, prob_cols=("A", "B", "C", "D"), ignored_label="-"):
        self.df = df.copy()
        self.prob_cols = list(prob_cols)
        self.ignored_label = ignored_label

        for c in self.prob_cols:
            self.df[c] = pd.to_numeric(self.df[c], errors="coerce")

        self.valid_mask = self.df[self.prob_cols].notna().all(axis=1)

        weights = np.array([4.0, 3.0, 2.0, 1.0], dtype=float)
        self.df["rank_score"] = np.nan
        self.df.loc[self.valid_mask, "rank_score"] = (
            self.df.loc[self.valid_mask, self.prob_cols].to_numpy(dtype=float) @ weights
        )

        self.df["assigned_grade"] = self.ignored_label

    def assign(self, class_number=None, percentages=None, labels=None):

        if percentages is None:
            if class_number is None:
                raise ValueError("Must specify either 'percentages' or 'class_number'.")
            percentages = [100 / class_number] * class_number
        else:
            class_number = len(percentages)

        if abs(sum(percentages) - 100) > 1e-5:
            raise ValueError(f"Percentages must sum to 100. Sum is {sum(percentages)}")

        if labels is None:
            labels = [string.ascii_uppercase[i] for i in range(class_number)]
        if len(labels) != class_number:
            raise ValueError(f"Number of labels ({len(labels)}) must match classes ({class_number}).")

        valid_df = self.df.loc[self.valid_mask].sort_values("rank_score", ascending=False)
        n = len(valid_df)
        if n == 0:
            return self.df["assigned_grade"]

        cumulative_pct = np.cumsum(percentages)
        cut_indices = [int(p / 100 * n) for p in cumulative_pct]
        cut_indices[-1] = n

        start = 0
        for end, label in zip(cut_indices, labels):
            idx = valid_df.index[start:end]
            self.df.loc[idx, "assigned_grade"] = label
            start = end

        return self.df["assigned_grade"]


# --- 분석 로직 쓰레드 ---

# --- 키워드 분류 ---
def keyword_classifier(df, target_col, keyword_dict, mode, extra_col=None, extra_dict=None, default=None):
    """
    Classifies observations based on keywords and optional extra column rules efficiently.

    Args:
        df (pd.DataFrame): The dataframe containing the data.
        target_col (str): The column name to search for keywords (e.g., 'text').
        keyword_dict (dict or pd.DataFrame): Dictionary or DataFrame mapping keywords to grades.
        mode (str): Conflict resolution mode - 'higher', 'lower', 'joint', or 'extra'
        extra_col (str): Optional column name for additional logic (e.g., '직위').
        extra_dict (dict): Optional dictionary mapping values in extra_col to grades.
        default (str): Default grade to assign if no keywords match.

    Returns:
        pd.Series: The calculated grades.
    """
    import re

    # tqdm은 개발환경에 따라 없을 수 있으므로 fallback 처리
    try:
        from tqdm import tqdm
    except Exception:
        def tqdm(x, **kwargs):
            return x

    if isinstance(keyword_dict, pd.DataFrame):
        kw_col = keyword_dict.columns[0]
        grade_col = keyword_dict.columns[1]

        duplicates = keyword_dict[keyword_dict.duplicated(subset=[kw_col], keep='first')]
        if not duplicates.empty:
            for key in duplicates[kw_col].unique():
                kept_grade = keyword_dict.loc[keyword_dict[kw_col] == key, grade_col].iloc[0]
                print(f"{key} 키워드가 중복됩니다. {key} 키워드는 먼저 입력한 {kept_grade}로 처리됩니다.")

        keyword_dict = keyword_dict.drop_duplicates(subset=[kw_col], keep='first')
        keyword_dict = dict(zip(keyword_dict[kw_col], keyword_dict[grade_col]))

    if mode == 'extra' and extra_dict is not None:
        if isinstance(extra_dict, pd.DataFrame):
            kw_col = extra_dict.columns[0]
            grade_col = extra_dict.columns[1]

            duplicates = extra_dict[extra_dict.duplicated(subset=[kw_col], keep='first')]
            if not duplicates.empty:
                for key in duplicates[kw_col].unique():
                    kept_grade = extra_dict.loc[extra_dict[kw_col] == key, grade_col].iloc[0]
                    print(f"{key} 직위가 중복됩니다. {key} 직위는 먼저 입력한 {kept_grade}로 처리됩니다.")

            extra_dict = extra_dict.drop_duplicates(subset=[kw_col], keep='first')
            extra_dict = dict(zip(extra_dict[kw_col], extra_dict[grade_col]))

    all_grades = set(keyword_dict.values()) if keyword_dict else set()
    if mode == 'extra' and extra_dict is not None:
        all_grades.update(extra_dict.values())
    unique_grades = sorted(list(all_grades))
    if not unique_grades:
        return pd.Series([default] * len(df), index=df.index)

    grade_to_rank = {g: i + 1 for i, g in enumerate(unique_grades)}
    rank_to_grade = {i + 1: g for i, g in enumerate(unique_grades)}

    grade_to_keywords = {}
    for k, v in keyword_dict.items():
        grade_to_keywords.setdefault(v, []).append(re.escape(str(k).replace(' ', '')))

    match_matrix = pd.DataFrame(index=df.index)

    target_series = df[target_col].astype(str).str.replace(' ', '', regex=False)

    for grade in tqdm(unique_grades, desc="Keyword Matching"):
        keywords = grade_to_keywords.get(grade, [])
        if keywords:
            pattern = '|'.join(keywords)
            match_matrix[grade] = target_series.str.contains(pattern, na=False)
        else:
            match_matrix[grade] = False

    rank_matrix = pd.DataFrame(index=df.index)
    for grade in unique_grades:
        rank = grade_to_rank[grade]
        rank_matrix[grade] = np.where(match_matrix[grade], rank, np.nan)

    if mode == 'joint':
        avg_rank = rank_matrix.mean(axis=1, skipna=True)
        final_rank = np.ceil(avg_rank)
    elif mode == 'higher':
        final_rank = rank_matrix.min(axis=1, skipna=True)
    elif mode == 'lower':
        final_rank = rank_matrix.max(axis=1, skipna=True)
    elif mode == 'extra':
        if not extra_col or not extra_dict:
            raise ValueError("extra_col and extra_dict are required for 'extra' mode.")

        # 키워드 기반 등급(기본 계산: 평균 후 올림)
        avg_rank = rank_matrix.mean(axis=1, skipna=True)
        keyword_rank = np.ceil(avg_rank)
        keyword_grade = keyword_rank.map(rank_to_grade)

        # "키워드 충돌" 마스크: 2개 이상 매칭된 행만 True
        conflict_mask = (match_matrix.sum(axis=1) > 1)

        # 직위 매핑 등급 (A~D 아니어도 문자열 그대로 허용)
        if extra_col in df.columns:
            extra_grade = df[extra_col].astype(str).map(extra_dict)
        else:
            extra_grade = pd.Series([np.nan] * len(df), index=df.index)

        extra_valid = extra_grade.notna() & (extra_grade.astype(str).str.strip() != "")

        final_grade = keyword_grade.copy()
        override_mask = conflict_mask & extra_valid
        final_grade[override_mask] = extra_grade[override_mask].astype(str).str.strip()

        # 아무 것도 없으면 default
        final_grade = final_grade.fillna(default)
        return final_grade

    else:
        raise ValueError(f"Unknown mode: {mode}")

    return final_rank.map(rank_to_grade).fillna(default)

class AnalysisThread(QThread):
    finished_signal = Signal(pd.DataFrame)
    error_signal = Signal(str)
    progress_signal = Signal(int)
    ask_ml_signal = Signal(int)

    def __init__(self, df, keyword_dict, mode, mapping, tokenizer, model, device, keyword_mode=None, ml_options=None, extra_dict=None):
        super().__init__()
        self.df = df
        self.keyword_dict = keyword_dict
        self.mode = mode
        self.mapping = mapping
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.ml_options = ml_options or {"strategy": "argmax"}
        self.run_ml_after = None 
        self.keyword_mode = keyword_mode
        self.extra_dict = extra_dict

    @torch.no_grad()
    def run(self):
        try:
            df = self.df.copy()
            self.progress_signal.emit(5)
            
            t_col = self.mapping.get('text_col', '')
            s4_col = self.mapping.get('stage4_col', '')
            s5_col = self.mapping.get('stage5_col', '')
            u_col = self.mapping.get('unit_col', '')
            p_col = self.mapping.get('extra_col', '')

            def combine_and_clean(row):
                parts = []
                if t_col in row and pd.notna(row[t_col]): parts.append(str(row[t_col]))
                for c in [s4_col, s5_col, u_col]:
                    if c in row and pd.notna(row[c]) and str(row[c]) != '없음':
                        parts.append(str(row[c]))
                text = " ".join([p.strip() for p in parts if p.strip()])
                return re.sub(r'\(.*?\)', '', text).strip()

            df['combined_input'] = df.apply(combine_and_clean, axis=1)
            for col in ['A', 'B', 'C', 'D']: df[col] = 0.0
            df['최종 예측 결과'] = "미분류"

            ml_suffix = "(머신러닝)" if self.mode == "키워드 방식" else ""

            if self.mode == "키워드 방식":
                if self.keyword_dict:
                    p_col = self.mapping.get('extra_col', '')

                    kw_series = keyword_classifier(
                        df,
                        "combined_input",
                        self.keyword_dict,
                        mode=self.keyword_mode,
                        extra_col=p_col if self.keyword_mode == "extra" else None,
                        extra_dict=self.extra_dict if self.keyword_mode == "extra" else None,
                        default=np.nan
                    )
                    matched_mask = kw_series.notna()
                    df.loc[matched_mask, "최종 예측 결과"] = kw_series[matched_mask].astype(str)

                    for g in ["A", "B", "C", "D"]:
                        df.loc[matched_mask & (df["최종 예측 결과"] == g), g] = 1.0
                self.progress_signal.emit(30)
                unclassified_mask = (df['최종 예측 결과'] == "미분류")
                unclassified_count = unclassified_mask.sum()
                
                if unclassified_count > 0:
                    self.ask_ml_signal.emit(int(unclassified_count))
                    while self.run_ml_after is None: self.msleep(100)
                else:
                    self.run_ml_after = False
            else:
                unclassified_mask = pd.Series([True] * len(df), index=df.index)
                self.run_ml_after = True
                self.progress_signal.emit(10)

            if self.run_ml_after and self.model:
                target_indices = df[unclassified_mask].index
                target_texts = df.loc[target_indices, 'combined_input'].tolist()
                
                batch_size = 8
                target_indices_list = list(target_indices)

                for i in range(0, len(target_texts), batch_size):
                    batch_texts = target_texts[i : i + batch_size]
                    batch_indices = target_indices_list[i : i + batch_size]

                    with torch.no_grad():
                        tokenized = self.tokenizer.batch_encode_plus(
                            batch_texts,
                            max_length=32,
                            padding=True,
                            truncation=True,
                            return_tensors="pt"
                        )
                        input_ids = tokenized["input_ids"].to(self.device)
                        attention_mask = tokenized["attention_mask"].to(self.device)

                        outputs = self.model(input_ids, attention_mask=attention_mask)
                        batch_probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()

                    for j, sub_idx in enumerate(batch_indices):
                        prob = batch_probs[j]

                        if p_col in df.columns:
                            pos_val = str(df.loc[sub_idx, p_col])
                            for pk, wv in POSITION_WEIGHTS.items():
                                if pk in pos_val:
                                    prob = prob * np.array(wv)
                                    break
                            if prob.sum() > 0:
                                prob = prob / prob.sum()

                        df.loc[sub_idx, ['A', 'B', 'C', 'D']] = prob
                        if self.ml_options.get("strategy", "argmax") == "argmax":
                            pred_grade = ['A', 'B', 'C', 'D'][np.argmax(prob)]
                            df.at[sub_idx, '최종 예측 결과'] = f"{pred_grade}{ml_suffix}"

                    prog = 30 + int(((i + len(batch_texts)) / len(target_texts)) * 65)
                    self.progress_signal.emit(min(prog, 95))

                # --- 분류 개수 지정 예측 ---
                if (self.mode != "키워드 방식") and (self.ml_options.get("strategy") == "quota"):
                    allocator = QuotaAllocator(df)
                    if self.ml_options.get("percentages") is not None:
                        df["최종 예측 결과"] = allocator.assign(percentages=self.ml_options["percentages"])
                    else:
                        df["최종 예측 결과"] = allocator.assign(class_number=self.ml_options.get("class_number"))
            if 'combined_input' in df.columns: df.drop(columns=['combined_input'], inplace=True)
            self.progress_signal.emit(100)
            self.finished_signal.emit(df)
        except Exception as e:
            self.error_signal.emit(str(e))

# --- UI 컴포넌트 ---
class ModeCard(QFrame):
    clicked = Signal(object)
    def __init__(self, title, description, icon_path, parent=None):
        super().__init__(parent)
        self.setFixedSize(350, 270) # 카드 가로세로
        self.selected = False
        self.title_text = title
        l = QVBoxLayout(self); l.setContentsMargins(20, 20, 20, 14); l.setSpacing(7)
        self.icon_lbl = QLabel(); self.icon_lbl.setAlignment(Qt.AlignCenter); self.icon_lbl.setFixedHeight(153)
        self.icon_lbl.setStyleSheet("background-color: #C2D6F9; border-radius: 10px; font-size: 50px;")

        # 이미지 로드 및 예외 처리 (없을 시 검정색)
        pm = hd_pixmap(self.icon_lbl, icon_path, 50)
        if not pm.isNull():
            # 이미지 크기를 레이블 크기에 맞춰 조절
            self.icon_lbl.setPixmap(pm)
        else:
            self.icon_lbl.setStyleSheet("background-color: #C2D6F9; border-radius: 10px; color: black;")
            self.icon_lbl.setText("No Image") 

        self.t_lbl = QLabel(title); self.t_lbl.setStyleSheet("font-size: 21px; font-weight: bold; color: #1F2937;")
        self.d_lbl = QLabel(description); self.d_lbl.setWordWrap(True); self.d_lbl.setStyleSheet("font-size: 13px; color: #4B5563;")
        l.addWidget(self.icon_lbl); l.addWidget(self.t_lbl); l.addWidget(self.d_lbl); l.addStretch()
        self.setCursor(QCursor(Qt.PointingHandCursor)); self.update_style()
    def update_style(self):
        c = "#2563EB" if self.selected else "#E5E7EB"
        self.setStyleSheet(f"ModeCard {{ background-color: white; border: 2px solid {c}; border-radius: 10px; }}")
    def mousePressEvent(self, e): self.clicked.emit(self)


# --- 머신러닝 예측 옵션 다이얼로그 ---
class MLOptionsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("머신러닝 예측 옵션")
        self.setModal(True)
        self.setFixedWidth(470)

        self.options = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(22, 20, 22, 18)
        layout.setSpacing(12)

        title = QLabel("머신러닝 예측 옵션")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #111827;")
        layout.addWidget(title)

        desc = QLabel("모형 실행 전에 예측 방식을 선택해 주세요.")
        desc.setStyleSheet("font-size: 13px; color: #6B7280;")
        layout.addWidget(desc)

        self.rb_argmax = QRadioButton("1) 4개 분류 예측 (A, B, C, D)  — 기본")
        self.rb_quota = QRadioButton("2) 분류 개수 지정 예측 (rank_score 기반)")
        self.rb_argmax.setChecked(True)

        layout.addWidget(self.rb_argmax)
        layout.addWidget(self.rb_quota)

        self.quota_frame = QFrame()
        self.quota_frame.setStyleSheet(
            "QFrame { background-color: #F9FAFB; border: 1px solid #E5E7EB; border-radius: 12px; }"
        )
        ql = QVBoxLayout(self.quota_frame)
        ql.setContentsMargins(14, 14, 14, 14)
        ql.setSpacing(8)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("분류 개수"))
        row1.addStretch()
        self.spin_classes = QSpinBox()
        self.spin_classes.setRange(2, 26)  # A~Z
        self.spin_classes.setValue(5)
        self.spin_classes.setFixedWidth(90)
        row1.addWidget(self.spin_classes)
        ql.addLayout(row1)

        self.rb_equal = QRadioButton("등급 비율 동일")
        self.rb_custom = QRadioButton("사용자 지정 비율")
        self.rb_equal.setChecked(True)
        ql.addWidget(self.rb_equal)
        ql.addWidget(self.rb_custom)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("비율(%)"))
        self.percent_edit = QLineEdit()
        self.percent_edit.setPlaceholderText("예: 5,15,20,20,40  (합계 100)")
        row2.addWidget(self.percent_edit)
        ql.addLayout(row2)

        layout.addWidget(self.quota_frame)

        btns = QHBoxLayout()
        btns.addStretch()
        self.btn_cancel = QPushButton("취소")
        self.btn_ok = QPushButton("확인")
        self.btn_cancel.setStyleSheet("QPushButton { background-color: #E5E7EB; color: #111827; border-radius: 10px; padding: 8px 18px; }")
        self.btn_ok.setStyleSheet("QPushButton { background-color: #2563EB; color: white; font-weight: bold; border-radius: 10px; padding: 8px 18px; }")
        btns.addWidget(self.btn_cancel)
        btns.addWidget(self.btn_ok)
        layout.addLayout(btns)

        self.btn_cancel.clicked.connect(self.reject)
        self.btn_ok.clicked.connect(self._on_ok)

        self.rb_quota.toggled.connect(self._update_enabled)
        self.rb_custom.toggled.connect(self._update_enabled)
        self._update_enabled()

    def _update_enabled(self):
        quota_on = self.rb_quota.isChecked()
        self.quota_frame.setEnabled(quota_on)
        self.percent_edit.setEnabled(quota_on and self.rb_custom.isChecked())

    def _on_ok(self):
        if self.rb_argmax.isChecked():
            self.options = {"strategy": "argmax"}
            self.accept()
            return

        class_number = int(self.spin_classes.value())

        if self.rb_equal.isChecked():
            self.options = {"strategy": "quota", "class_number": class_number, "percentages": None}
            self.accept()
            return

        raw = self.percent_edit.text().strip()
        if not raw:
            QMessageBox.warning(self, "입력 오류", "사용자 지정 비율을 입력해 주세요.")
            return
        try:
            parts = [float(x.strip()) for x in raw.split(",") if x.strip() != ""]
        except Exception:
            QMessageBox.warning(self, "입력 오류", "비율은 쉼표로 구분된 숫자로 입력해 주세요. 예: 5,15,20,20,40")
            return

        if len(parts) != class_number:
            QMessageBox.warning(self, "입력 오류", f"비율 개수({len(parts)})가 분류 개수({class_number})와 같아야 합니다.")
            return

        if abs(sum(parts) - 100.0) > 1e-5:
            QMessageBox.warning(self, "입력 오류", f"비율 합계가 100이 되어야 합니다. 현재 합계: {sum(parts)}")
            return

        self.options = {"strategy": "quota", "class_number": class_number, "percentages": parts}
        self.accept()

    def get_options(self):
        return self.options

# --- 키워드기반 예측 옵션 다이얼로그 ---
class KeywordOptionsDialog(QDialog):
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("중복 키워드 처리 옵션")
        self.setModal(True)
        self.setFixedWidth(470)

        self.options = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(22, 20, 22, 18)
        layout.setSpacing(12)

        title = QLabel("중복 키워드 처리 옵션")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #111827;")
        layout.addWidget(title)

        desc = QLabel("중복된 키워드에 대해 어떻게 처리할까요?")
        desc.setStyleSheet("font-size: 13px; color: #6B7280;")
        layout.addWidget(desc)

        self.rb_higher = QRadioButton("1) 높은 등급")
        self.rb_lower = QRadioButton("2) 낮은 등급")
        self.rb_joint = QRadioButton("3) 평균 등급")
        self.rb_extra = QRadioButton("4) 직위 기준")

        layout.addWidget(self.rb_higher)
        layout.addWidget(self.rb_lower)
        layout.addWidget(self.rb_joint)
        layout.addWidget(self.rb_extra)

        btns = QHBoxLayout()
        btns.addStretch()
        self.btn_cancel = QPushButton("취소")
        self.btn_ok = QPushButton("확인")
        self.btn_cancel.setStyleSheet("QPushButton { background-color: #E5E7EB; color: #111827; border-radius: 10px; padding: 8px 18px; }")
        self.btn_ok.setStyleSheet("QPushButton { background-color: #2563EB; color: white; font-weight: bold; border-radius: 10px; padding: 8px 18px; }")
        btns.addWidget(self.btn_cancel)
        btns.addWidget(self.btn_ok)
        layout.addLayout(btns)

        self.btn_cancel.clicked.connect(self.reject)
        self.btn_ok.clicked.connect(self._on_ok)

    def _on_ok(self):
        if self.rb_higher.isChecked():
            self.options = "higher"
            self.accept()
            return
        
        if self.rb_lower.isChecked():
            self.options = "lower"
            self.accept()
            return
        
        if self.rb_joint.isChecked():
            self.options = "joint"
            self.accept()
            return
        
        if self.rb_extra.isChecked():
            self.options = "extra"
            self.accept()
            return
        
    def get_options(self):
        return self.options

class PositionMappingDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("직위 기준 매핑 입력")
        self.setModal(True)
        self.resize(560, 420)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 14)
        layout.setSpacing(10)

        title = QLabel("직위-등급 매핑표")
        title.setStyleSheet("font-size: 18px; font-weight: 700; color: #111827;")
        layout.addWidget(title)

        desc = QLabel("표에 직접 입력하거나 엑셀/CSV 업로드, 또는 엑셀에서 복사(Ctrl+C) → 여기 붙여넣기(Ctrl+V) 가능")
        desc.setStyleSheet("font-size: 13px; color: #6B7280;")
        layout.addWidget(desc)

        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["직위", "등급"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.DoubleClicked | QAbstractItemView.SelectedClicked | QAbstractItemView.EditKeyPressed)
        self.table.setStyleSheet("""
            QTableWidget { border: 1px solid #E5E7EB; border-radius: 10px; gridline-color: #E5E7EB; }
            QHeaderView::section { background: #F9FAFB; padding: 8px; border: none; border-bottom: 1px solid #E5E7EB; font-weight: 600; }
        """)
        layout.addWidget(self.table)

        row = QHBoxLayout()
        self.btn_add = QPushButton("＋ 행 추가")
        self.btn_del = QPushButton("선택 삭제")
        self.btn_load = QPushButton("엑셀/CSV 업로드")

        for b in (self.btn_add, self.btn_del, self.btn_load):
            b.setStyleSheet("QPushButton { background-color: #E5E7EB; color: #111827; border-radius: 10px; padding: 8px 12px; }")

        self.btn_add.clicked.connect(self.add_row)
        self.btn_del.clicked.connect(self.delete_rows)
        self.btn_load.clicked.connect(self.load_file)

        row.addWidget(self.btn_add)
        row.addWidget(self.btn_del)
        row.addStretch()
        row.addWidget(self.btn_load)
        layout.addLayout(row)

        btns = QHBoxLayout()
        btns.addStretch()
        self.btn_cancel = QPushButton("취소")
        self.btn_ok = QPushButton("적용")
        self.btn_cancel.setStyleSheet("QPushButton { background-color: #E5E7EB; color: #111827; border-radius: 10px; padding: 8px 18px; }")
        self.btn_ok.setStyleSheet("QPushButton { background-color: #2563EB; color: white; font-weight: 700; border-radius: 10px; padding: 8px 18px; }")
        self.btn_cancel.clicked.connect(self.reject)
        self.btn_ok.clicked.connect(self.accept_if_valid)
        btns.addWidget(self.btn_cancel)
        btns.addWidget(self.btn_ok)
        layout.addLayout(btns)

        for _ in range(5):
            self.add_row()

        self.table.installEventFilter(self)

    def add_row(self):
        r = self.table.rowCount()
        self.table.insertRow(r)
        self.table.setItem(r, 0, QTableWidgetItem(""))
        self.table.setItem(r, 1, QTableWidgetItem(""))

    def delete_rows(self):
        rows = sorted({idx.row() for idx in self.table.selectionModel().selectedRows()}, reverse=True)
        for r in rows:
            self.table.removeRow(r)

    def eventFilter(self, obj, event):
        if obj is self.table and event.type() == QEvent.KeyPress:
            if event.matches(QKeySequence.Paste):
                self.paste_from_clipboard()
                return True
        return super().eventFilter(obj, event)

    def paste_from_clipboard(self):
        text = QApplication.clipboard().text()
        if not text.strip():
            return

        start_row = self.table.currentRow()
        if start_row < 0:
            start_row = self.table.rowCount()

        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        r = start_row
        for ln in lines:
            if "\t" in ln:
                a, b = ln.split("\t", 1)
            elif "," in ln:
                a, b = ln.split(",", 1)
            else:
                parts = ln.split()
                if len(parts) < 2:
                    continue
                a, b = parts[0], parts[1]

            if r >= self.table.rowCount():
                self.add_row()

            self.table.item(r, 0).setText(str(a).strip())
            self.table.item(r, 1).setText(str(b).strip())
            r += 1

    def load_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "직위 매핑 파일 열기", "", "Excel/CSV (*.xlsx *.xls *.csv)")
        if not path:
            return
        try:
            if path.lower().endswith(".csv"):
                try:
                    df = pd.read_csv(path, encoding="utf-8-sig")
                except Exception:
                    df = pd.read_csv(path, encoding="cp949")
            else:
                df = pd.read_excel(path)

            if df is None or df.empty:
                raise ValueError("파일이 비어 있습니다.")

            df.columns = [str(c).strip() for c in df.columns]

            pos_col = next((c for c in ["직위", "직위명", "position", "Position"] if c in df.columns), None)
            grd_col = next((c for c in ["등급", "grade", "Grade", "단계"] if c in df.columns), None)
            if pos_col is None or grd_col is None:
                if len(df.columns) < 2:
                    raise ValueError("최소 2개 컬럼(직위/등급)이 필요합니다.")
                pos_col, grd_col = df.columns[0], df.columns[1]

            self.table.setRowCount(0)
            count = 0
            for _, row in df.iterrows():
                p = "" if pd.isna(row[pos_col]) else str(row[pos_col]).strip()
                g = "" if pd.isna(row[grd_col]) else str(row[grd_col]).strip()
                if not p or not g:
                    continue
                self.add_row()
                r = self.table.rowCount() - 1
                self.table.item(r, 0).setText(p)
                self.table.item(r, 1).setText(g)
                count += 1

            if self.table.rowCount() == 0:
                for _ in range(5):
                    self.add_row()

            QMessageBox.information(self, "완료", f"{count}개 항목을 불러왔습니다.")
        except Exception as e:
            QMessageBox.critical(self, "오류", f"파일을 불러오는 중 오류:\n{str(e)}")

    def accept_if_valid(self):
        m = self.get_mapping()
        if not m:
            QMessageBox.warning(self, "입력 오류", "최소 1개 이상 (직위, 등급)을 입력해 주세요.")
            return
        self.accept()

    def get_mapping(self) -> dict:
        mapping = {}
        for r in range(self.table.rowCount()):
            p_item = self.table.item(r, 0)
            g_item = self.table.item(r, 1)
            p = "" if p_item is None else str(p_item.text()).strip()
            g = "" if g_item is None else str(g_item.text()).strip()
            if not p or not g:
                continue
            if p not in mapping:
                mapping[p] = g
        return mapping


# --- 메인 윈도우 ---
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("문서 처리 등급 분류 도구")
        self.setWindowIcon(QIcon(resource_path("assets/icon.ico")))
        self.resize(1200, 700)

        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None; self.model = None
        self.current_df = None; self.final_df = None; self.selected_mode = None
        self.mapped_dept_col = "없음"
        
        self.init_model()
        
        self.central_stacked = QStackedWidget()
        self.setCentralWidget(self.central_stacked)
        
        self.setup_init_screen()
        self.setup_main_work_screen()
        
        self.central_stacked.addWidget(self.init_screen)
        self.central_stacked.addWidget(self.main_work_screen)
        self.extra_dict = None
        self._kw_default_seeded = False

    def init_model(self):
        model_path = None
        model_path = resource_path("model")

        if model_path is None:
            print("모델 폴더를 찾지 못했습니다.")
            return

        try:
            # 모델 로드
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_path, num_labels=4
            ).to(self.device)

            vocab_file = os.path.join(model_path, "tokenizer_78b3253a26.model")
            vocab_txt = os.path.join(model_path, "vocab.txt")

            if os.path.exists(vocab_file) and os.path.exists(vocab_txt):
                self.tokenizer = KoBertTokenizer(vocab_file=vocab_file, vocab_txt=vocab_txt)
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)

            ckpt = os.path.join(model_path, "best_model_weights.pth")
            if os.path.exists(ckpt):
                try:
                    self.model.load_state_dict(
                        torch.load(ckpt, map_location=self.device, weights_only=True),
                        strict=False
                    )
                except TypeError:
                    self.model.load_state_dict(
                        torch.load(ckpt, map_location=self.device),
                        strict=False
                    )

            self.model.eval()
            print(f"모델 로드 완료: {model_path}")

        except Exception as e:
            print(f"모델 로드 오류: {e}")
            self.tokenizer = None
            self.model = None

    def colored_icon(self, ref_widget, path, color_hex):
        size = ref_widget.iconSize().width() or 17
        pm = hd_pixmap(ref_widget, path, size)
        if pm.isNull():
            return QIcon()

        colored = QPixmap(pm.size())
        colored.setDevicePixelRatio(pm.devicePixelRatioF())
        colored.fill(Qt.transparent)

        p = QPainter(colored)
        p.drawPixmap(0, 0, pm)
        p.setCompositionMode(QPainter.CompositionMode_SourceIn)
        p.fillRect(colored.rect(), QColor(color_hex))
        p.end()

        return QIcon(colored)

    def setup_init_screen(self):
        self.init_screen = QWidget(); layout = QVBoxLayout(self.init_screen); layout.setContentsMargins(50, 70, 50, 40)
        header = QLabel("문서 처리 등급 분류 도구"); header.setAlignment(Qt.AlignCenter); header.setStyleSheet("font-size: 36px; font-weight: bold; color: #111827; margin-bottom: 8px;")
        sub_header = QLabel("원하는 분류 방식을 선택하여 문서 등급 산정을 시작하십시오."); sub_header.setAlignment(Qt.AlignCenter); sub_header.setStyleSheet("font-size: 16px; color: #6B7280; font-weight: bold; color: #494949; margin-bottom: 40px;")
        
        cards_layout = QHBoxLayout()
        self.card_keyword = ModeCard("키워드 방식", "사전에 정의된 핵심 키워드 목록을 기반으로 문서의 등급을 신속하게 분류합니다.", SEARCH_ICON_PATH)
        self.card_ml = ModeCard("머신러닝 방식", "AI 모델을 활용하여 문서의 전체적인 맥락과 의미를 분석하고 고도화된 분류 결과를 제공합니다.", ML_ICON_PATH)
        self.card_keyword.clicked.connect(self.select_card); self.card_ml.clicked.connect(self.select_card)
        cards_layout.addStretch(); cards_layout.addWidget(self.card_keyword); cards_layout.addSpacing(60); cards_layout.addWidget(self.card_ml); cards_layout.addStretch()
        self.start_btn = QPushButton("시작 →"); self.start_btn.setFixedSize(240, 90); self.start_btn.setEnabled(False)
        self.start_btn.setStyleSheet("QPushButton { background-color: #2563EB; color: white; font-size: 18px; font-weight: bold; border-radius: 12px; margin-top: 40px; } QPushButton:disabled { background-color: #D1D1D1; }")
        self.start_btn.clicked.connect(self.go_to_config)
        layout.addWidget(header); layout.addWidget(sub_header); layout.addLayout(cards_layout); layout.addWidget(self.start_btn, alignment=Qt.AlignCenter); layout.addStretch()

    def setup_main_work_screen(self):
        self.main_work_screen = QWidget()
        main_h = QHBoxLayout(self.main_work_screen); main_h.setContentsMargins(0, 0, 0, 0); main_h.setSpacing(0)

        self.sidebar = QFrame(); self.sidebar.setFixedWidth(220)
        self.sidebar.setStyleSheet("background-color: #FFFFFF; border-right: 1px solid #F3F3F3;")
        side_v = QVBoxLayout(self.sidebar); side_v.setContentsMargins(10, 30, 10, 20)

        app_row = QHBoxLayout()
        app_row.setContentsMargins(15, 0, 0, 0)
        app_row.setSpacing(8)


        app_icon = QLabel()
        app_icon.setPixmap(hd_pixmap(app_icon, TITLE_ICON_PATH, 32))
        app_icon.setFixedSize(32, 32)
        app_icon.setAlignment(Qt.AlignVCenter)

        app_info = QLabel("공공문서 관리")
        app_info.setStyleSheet("font-weight: bold; color: #1F2937; font-size: 17px;")
        app_info.setAlignment(Qt.AlignVCenter)

        app_row.addWidget(app_icon, 0, Qt.AlignVCenter)
        app_row.addWidget(app_info, 0, Qt.AlignVCenter)

        side_v.addLayout(app_row)
        side_v.addSpacing(25)

        # 메뉴 버튼 생성
        self.menu_predict = QPushButton(" 문서 관리")
        self.menu_predict.setIconSize(QSize(18, 18))

        self.menu_dashboard = QPushButton(" 대시보드")
        self.menu_dashboard.setIconSize(QSize(18, 18))
        
        self.menu_buttons = [self.menu_predict, self.menu_dashboard]

        for btn in self.menu_buttons:
            btn.setFixedHeight(60)
            side_v.addWidget(btn)

        # 클릭 이벤트 연결 및 초기 스타일 설정
        self.menu_predict.clicked.connect(lambda: self.change_menu(0))
        self.menu_dashboard.clicked.connect(lambda: self.change_menu(1))

        side_v.addStretch()

        reset_btn = QPushButton("↩ 처음 화면으로"); reset_btn.setFixedHeight(40)
        reset_btn.setStyleSheet("""
            QPushButton { background-color: #EBF2FB; color: #4B5563; border: 5px; border-radius: 5px; }
            QPushButton:hover { background-color: #C2D6F9; }
        """)
        reset_btn.clicked.connect(lambda: self.central_stacked.setCurrentIndex(0))
        side_v.addWidget(reset_btn)

        self.content_stack = QStackedWidget()
        self.predict_page = self.setup_predict_ui()
        self.dashboard_page = self.setup_dashboard_ui()
        self.result_page = self.setup_result_ui()
        
        self.content_stack.addWidget(self.predict_page) 
        self.content_stack.addWidget(self.dashboard_page)
        self.content_stack.addWidget(self.result_page)

        main_h.addWidget(self.sidebar)
        main_h.addWidget(self.content_stack)

        self.update_menu_style(self.menu_predict)

    def update_menu_style(self, active_button):
        for btn in self.menu_buttons:
            if btn == active_button:
                # 활성화 상태: 연한 파랑 배경, 굵은 파랑 글씨
                btn.setStyleSheet("""
                    QPushButton {
                        text-align: left; 
                        padding-left: 20px; 
                        border: none; 
                        border-radius: 12px; 
                        font-size: 15px; 
                        font-weight: bold;
                        background-color: #EBF2FB; 
                        color: #537FDE;
                    }
                """)
                if btn == self.menu_predict:
                    btn.setIcon(self.colored_icon(btn, DOC_ICON_PATH, "#537FDE"))
                else:
                    btn.setIcon(self.colored_icon(btn, DASH_ICON_PATH, "#537FDE"))
            else:
                # 비활성화 상태: 투명 배경, 회색 글씨
                btn.setStyleSheet("""
                    QPushButton {
                        text-align: left; 
                        padding-left: 20px; 
                        border: none; 
                        border-radius: 12px; 
                        font-size: 15px; 
                        color: #A1AABC; 
                        background-color: transparent;
                    }
                    QPushButton:hover { 
                        background-color: #F3F4F6; 
                    }
                """)
                if btn == self.menu_predict:
                    btn.setIcon(self.colored_icon(btn, DOC_ICON_PATH, "#A1AABC"))
                else:
                    btn.setIcon(self.colored_icon(btn, DASH_ICON_PATH, "#A1AABC"))

    def change_menu(self, index):
        """페이지 전환"""
        if index == 1:
            self.update_and_show_dashboard()
        else:
            self.content_stack.setCurrentIndex(index)
            self.update_menu_style(self.menu_predict)

    def setup_predict_ui(self):
        page = QWidget(); layout = QVBoxLayout(page); layout.setContentsMargins(20, 20, 20, 20)
        top = QHBoxLayout()
        self.config_title = QLabel("데이터 분석 설정"); self.config_title.setStyleSheet("color: #1F2937; font-size: 22px; font-weight: bold;")
        self.rb_cp949 = QRadioButton("CP949(CSV)"); self.rb_utf8 = QRadioButton("UTF-8(Excel)"); self.rb_utf8.setChecked(True)

        # 설정 내보내기 버튼
        self.export_settings_btn = QPushButton("직위 키워드 내보내기")
        self.export_settings_btn.setFixedHeight(30)
        self.export_settings_btn.setStyleSheet("""
            QPushButton { background-color: #C9CFDD; color: #1F2937; border-radius: 10px; padding: 6px 12px; }
            QPushButton:hover { background-color: #A1AABC; }
        """)
        self.export_settings_btn.clicked.connect(self.export_position_mapping)

        top.addWidget(self.config_title)
        top.addStretch()
        top.addWidget(self.export_settings_btn)
        top.addSpacing(8)
        top.addWidget(self.rb_cp949)
        top.addWidget(self.rb_utf8)
        
        mid = QHBoxLayout()
        self.kw_container = QWidget(); self.kw_container.setFixedWidth(330)
        kw_v = QVBoxLayout(self.kw_container); kw_v.setContentsMargins(0,0,10,0)
        kw_header_v = QVBoxLayout()
        kw_header_v.setSpacing(6)

        top_row = QHBoxLayout()
        top_row.addWidget(QLabel(" 키워드별 등급 설정:"))

        self.grade_order_edit = QLineEdit("A,B,C,D")
        self.grade_order_edit.setPlaceholderText("등급 목록 (예: A,B,C,D,E,F)")
        self.grade_order_edit.setFixedWidth(200)
        self.grade_order_edit.editingFinished.connect(self.refresh_grade_comboboxes)
        top_row.addWidget(self.grade_order_edit)

        top_row.addStretch()
        kw_header_v.addLayout(top_row)

        bottom_row = QHBoxLayout()
        bottom_row.addStretch()


        # 키워드 내보내기 버튼
        export_kw_btn = QPushButton("키워드 내보내기")
        export_kw_btn.setFixedWidth(100)
        export_kw_btn.clicked.connect(self.export_keywords)
        bottom_row.addWidget(export_kw_btn)

        load_kw_btn = QPushButton("키워드 업로드")
        load_kw_btn.setFixedWidth(100)
        load_kw_btn.clicked.connect(self.load_keywords_file)
        bottom_row.addWidget(load_kw_btn)

        add_btn = QPushButton("+ 추가")
        del_btn = QPushButton("- 삭제")
        for b in [add_btn, del_btn]:
            b.setFixedWidth(55)

        add_btn.clicked.connect(self.add_kw_row)
        del_btn.clicked.connect(self.del_kw_row)
        bottom_row.addWidget(add_btn)
        bottom_row.addWidget(del_btn)
        kw_header_v.addLayout(bottom_row)

        
        self.kw_table = QTableWidget(0, 2); self.kw_table.setHorizontalHeaderLabels(["키워드", "등급"])
        self.kw_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        kw_v.addLayout(kw_header_v); kw_v.addWidget(self.kw_table)
        
        pre_v = QVBoxLayout()
        pre_v.addWidget(QLabel("데이터 미리보기:"))
        self.preview_table = QTableWidget(); pre_v.addWidget(self.preview_table)
        
        mid.addWidget(self.kw_container); mid.addLayout(pre_v, stretch=1)
        
        btns = QHBoxLayout()
        self.upload_btn = QPushButton("파일 불러오기"); self.upload_btn.setFixedHeight(45); self.upload_btn.clicked.connect(self.load_file)
        self.analyze_btn = QPushButton("분석 및 예측 시작"); self.analyze_btn.setFixedHeight(45); self.analyze_btn.setEnabled(False)
        self.upload_btn.setStyleSheet("""
            QPushButton {background-color: #FFFFFF; color: black; font-weight: bold; border-radius: 8px; }
            QPushButton:hover {background-color: #F9F9F9;}
        """)
        self.analyze_btn.setStyleSheet("""
            QPushButton { background-color: #2563EB; color: white; font-weight: bold; border-radius: 8px; }
            QPushButton:disabled { background-color: #C2D6F9; color: white; }
        """)
        self.analyze_btn.clicked.connect(self.show_mapping_dialog)
        btns.addWidget(self.upload_btn); btns.addWidget(self.analyze_btn)
        
        layout.addLayout(top); layout.addLayout(mid); layout.addLayout(btns)
        return page

    def setup_dashboard_ui(self):
        page = QWidget(); layout = QVBoxLayout(page); layout.setContentsMargins(20, 20, 20, 20)
        layout.addWidget(QLabel("문서 분류 요약 현황", styleSheet="font-size: 24px; font-weight: bold; margin-bottom: 15px;"))
        
        layout.addWidget(QLabel("전체 부서명별 등급 분포"))
        card = QFrame()
        card.setAttribute(Qt.WA_StyledBackground, True)
        card.setStyleSheet("QFrame{background:#fff;border:1px solid #E5E7EB;border-radius:10px;}")

        card_l = QVBoxLayout(card)
        card_l.setContentsMargins(2, 2, 2, 2)

        self.stats_table = QTableWidget()
        self.stats_table.setFrameShape(QFrame.NoFrame)
        self.stats_table.setStyleSheet("QTableWidget{background:transparent;border:none;gridline-color: #E5E7EB;}")

        card_l.addWidget(self.stats_table)
        layout.addWidget(card)

        card_h = QHBoxLayout()
        self.card_top_dept = self.create_stat_card("최다 분류 부서", "-", "")
        self.card_a_ratio = self.create_stat_card("A등급 비율", "0%", "")
        self.card_total = self.create_stat_card("총 문서 분류량", "0건", "")
        self.card_pending = self.create_stat_card("미분류 문서", "0건", "")
        card_h.addWidget(self.card_top_dept); card_h.addWidget(self.card_a_ratio); card_h.addWidget(self.card_total); card_h.addWidget(self.card_pending)
        layout.addLayout(card_h)
        return page

    def setup_result_ui(self):
        page = QWidget(); layout = QVBoxLayout(page)
        self.res_table = QTableWidget()
        save_btn = QPushButton("결과 엑셀 저장"); save_btn.setFixedHeight(50); 
        save_btn.setStyleSheet("""
            QPushButton { background-color: #2563EB; color: white; font-weight: bold; border-radius: 10px; }
            QPushButton:hover { background-color: #C2D6F9; }                   
        """) 
        save_btn.clicked.connect(self.save_excel)
        layout.addWidget(QLabel("최종 분석 결과", styleSheet="font-size: 20px; font-weight: bold;")); layout.addWidget(self.res_table); layout.addWidget(save_btn)
        return page

    def create_stat_card(self, title, val, sub):
        card = QFrame()
        card.setStyleSheet("""
            QFrame { background-color: white; border: 1px solid #E5E7EB; border-radius: 10px; padding: 10px; }
        """)
        v = QVBoxLayout(card); v.setSpacing(2)
        
        t_l = QLabel(title)
        t_l.setStyleSheet("color: #6B7280; font-size: 13px; border: none; background: transparent;")
        
        v_l = QLabel(val); v_l.setWordWrap(True)
        v_l.setStyleSheet("font-size: 18px; font-weight: bold; color: #111827; border: none; background: transparent;")
        
        v.addWidget(t_l); v.addWidget(v_l); v.addStretch()
        
        card.val_label = v_l
        return card
    
    def update_and_show_dashboard(self):
        if self.final_df is None:
            QMessageBox.warning(self, "알림", "분석된 데이터가 없습니다. 먼저 분석을 완료해 주세요.")
            return
        else :
            self.update_menu_style(self.menu_dashboard)
            df = self.final_df.copy()
            res_col = "최종 예측 결과"

            dept_col = self.mapped_dept_col
            if dept_col == "없음" or dept_col not in df.columns:
                dept_col = df.columns[0]

            # 데이터 집계 (부서별 등급 개수)
            df["temp_grade"] = df[res_col].astype(str).str.strip().str[0]

            summary = df.groupby([dept_col, "temp_grade"]).size().unstack(fill_value=0)

            if "미" in summary.columns:
                summary = summary.rename(columns={"미": "미분류"})

            # 등급 컬럼 목록 동적으로 구성 (합계 제외)
            grade_cols = [c for c in summary.columns if c != "합계"]

            # 정렬: A,B,C... 먼저, 그 외(미분류 등)는 뒤로
            def grade_sort_key(x):
                x = str(x)
                if len(x) == 1 and x.isalpha():
                    return (0, ord(x.upper()) - ord("A"))
                if x == "미분류":
                    return (2, 999)
                return (1, x)

            grade_cols = sorted(grade_cols, key=grade_sort_key)

            # 합계 컬럼 추가
            summary["합계"] = summary[grade_cols].sum(axis=1) if grade_cols else 0

            # ---------- 테이블 설정 ----------
            headers = ["부서명"] + [f"{g}등급" if g != "미분류" else "미분류" for g in grade_cols] + ["합계"]
            self.stats_table.setRowCount(len(summary) + 1)  # + 전체 합계 행
            self.stats_table.setColumnCount(len(headers))
            self.stats_table.setHorizontalHeaderLabels(headers)

            # 1) 일반 부서 데이터 삽입
            for i, (dept, row) in enumerate(summary.iterrows()):
                self.stats_table.setItem(i, 0, QTableWidgetItem(str(dept)))
                for j, g in enumerate(grade_cols):
                    self.stats_table.setItem(i, j + 1, QTableWidgetItem(str(int(row.get(g, 0)))))
                self.stats_table.setItem(i, len(headers) - 1, QTableWidgetItem(str(int(row.get("합계", 0)))))

            # 2) 전체 합계 행 계산 및 삽입
            total_row_idx = len(summary)
            total_count = int(summary["합계"].sum())

            item_total_label = QTableWidgetItem("전체 합계")
            font = item_total_label.font(); font.setBold(True); item_total_label.setFont(font)
            
            item_total_label.setForeground(QColor("#2563EB"))
            self.stats_table.setItem(total_row_idx, 0, item_total_label)

            # 등급별 합계 + 비율 표시
            for j, g in enumerate(grade_cols):
                grade_sum = int(summary[g].sum()) if g in summary.columns else 0
                percentage = (grade_sum / total_count * 100) if total_count > 0 else 0
                display_text = f"{grade_sum} ({percentage:.1f}%)"

                item = QTableWidgetItem(display_text)
                font = item.font(); font.setBold(True); item.setFont(font)
                item.setForeground(QColor("#2563EB"))
                self.stats_table.setItem(total_row_idx, j + 1, item)

            # 총합(문서 수)
            item_total_all = QTableWidgetItem(str(total_count))
            font = item_total_all.font(); font.setBold(True); item_total_all.setFont(font)
            item_total_all.setForeground(QColor("#2563EB"))
            self.stats_table.setItem(total_row_idx, len(headers) - 1, item_total_all)

            self.stats_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

            # ---------- 하단 카드 요약 정보 업데이트 ----------
            self.card_total.val_label.setText(f"{total_count}건")

            # 최상위 등급 비율: 알파벳 등급이 있으면 가장 앞(보통 A)
            top_grade = None
            for g in grade_cols:
                if len(str(g)) == 1 and str(g).isalpha():
                    top_grade = g
                    break
            if top_grade is not None and total_count > 0:
                self.card_a_ratio.val_label.setText(f"{(summary[top_grade].sum() / total_count * 100):.1f}%")
            else:
                self.card_a_ratio.val_label.setText("0%")

            # 최다 분류 부서
            if len(summary) > 0:
                self.card_top_dept.val_label.setText(f"{summary['합계'].idxmax()}")
            else:
                self.card_top_dept.val_label.setText("-")

            # 미분류 문서 수
            pending = int(summary["미분류"].sum()) if "미분류" in summary.columns else 0
            self.card_pending.val_label.setText(f"{pending}건")

            self.content_stack.setCurrentIndex(1)
            return

    def select_card(self, card):
        self.card_keyword.selected = self.card_ml.selected = False; card.selected = True
        self.card_keyword.update_style(); self.card_ml.update_style(); self.start_btn.setEnabled(True)
        self.selected_mode = card.title_text

    def go_to_config(self):
        if self.selected_mode == "키워드 방식":
            self.config_title.setText(" 키워드 기반 예측")
            self.kw_container.setVisible(True)
            self.seed_default_keyword_rows()
        else:
            self.config_title.setText(" 머신러닝 기반 예측")
            self.kw_container.setVisible(False)
        self.central_stacked.setCurrentIndex(1)
        self.content_stack.setCurrentIndex(0)

    def load_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "파일 열기", "", "Excel/CSV (*.xlsx *.csv)")
        if not path: return
        
        try:
            # 인코딩 설정
            if path.endswith('.csv'):
                enc = "cp949" if self.rb_cp949.isChecked() else "utf-8-sig"
                df = pd.read_csv(path, encoding=enc, engine='python', sep=None, on_bad_lines='skip')
            else:
                df = pd.read_excel(path)

            if df is not None:
                # 컬럼명 공백 제거
                df.columns = [str(c).strip() for c in df.columns]

                # 괄호 내용 제거 여부 확인
                reply = QMessageBox.question(
                    self, "데이터 전처리", 
                    "불러온 데이터에서 소괄호 ( ) 안의 내용을 모두 삭제하시겠습니까?",
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No
                )

                if reply == QMessageBox.Yes:
                    df = df.applymap(lambda x: re.sub(r'\(.*?\)', '', str(x)).strip() if pd.notnull(x) else x)
                    df.columns = [re.sub(r'\(.*?\)', '', c).strip() for c in df.columns]

                self.current_df = df
                self.display_df_on_table(self.preview_table, self.current_df.head(50))
                
                # 버튼 상태 및 결과 데이터 초기화
                self.final_df = None
                self.analyze_btn.setEnabled(True)
                self.analyze_btn.setText("분석 및 예측 시작")
                self.analyze_btn.setStyleSheet("""
                    QPushButton { background-color: #2563EB; color: white; font-weight: bold; border-radius: 8px; }
                    QPushButton:hover { background-color: #234DAA; }
                """)
                
                try: self.analyze_btn.clicked.disconnect()
                except: pass
                self.analyze_btn.clicked.connect(self.show_mapping_dialog)
                
                msg = "괄호 제거 후 로드 완료" if reply == QMessageBox.Yes else "데이터 로드 완료"
                QMessageBox.information(self, "성공", f"{len(self.current_df)}건 {msg}")
                
        except Exception as e:
            QMessageBox.critical(self, "오류", f"파일을 불러오는 중 오류가 발생했습니다:\n{str(e)}")

    def display_df_on_table(self, table_widget, df):
        if df is None or df.empty: return
        table_widget.setRowCount(len(df)); table_widget.setColumnCount(len(df.columns))
        table_widget.setHorizontalHeaderLabels(df.columns)
        for i in range(len(df)):
            for j in range(len(df.columns)):
                v = df.iloc[i, j]
                table_widget.setItem(i, j, QTableWidgetItem(str(v) if pd.notnull(v) else ""))

    def get_grade_list(self):
        """등급 목록 입력창(QLineEdit)에서 등급 리스트를 가져옵니다. (예: A,B,C,D,E,F)"""
        if not hasattr(self, "grade_order_edit") or self.grade_order_edit is None:
            return ["A", "B", "C", "D"]
        raw = self.grade_order_edit.text().strip()
        grades = [g.strip() for g in raw.split(",") if g.strip()]
        return grades if grades else ["A", "B", "C", "D"]

    def refresh_grade_comboboxes(self):
        grades = self.get_grade_list()
        for r in range(self.kw_table.rowCount()):
            cb = self.kw_table.cellWidget(r, 1)
            if isinstance(cb, QComboBox):
                cur = cb.currentText()
                cb.blockSignals(True)
                cb.clear()
                cb.setEditable(True)
                cb.addItems(grades)
                cb.setCurrentText(cur)
                cb.blockSignals(False)

    def add_kw_row(self):
        r = self.kw_table.rowCount()
        self.kw_table.insertRow(r)
        self.kw_table.setItem(r, 0, QTableWidgetItem(""))
        cb = QComboBox()
        cb.setEditable(True)
        cb.addItems(self.get_grade_list())
        self.kw_table.setCellWidget(r, 1, cb)

    def seed_default_keyword_rows(self):
        if getattr(self, "_kw_default_seeded", False):
            return
        if not hasattr(self, "kw_table") or self.kw_table is None:
            return
        if self.kw_table.rowCount() > 0:
            self._kw_default_seeded = True
            return

        for g in ["A", "B", "C", "D"]:
            self.add_kw_row()
            r = self.kw_table.rowCount() - 1
            cb = self.kw_table.cellWidget(r, 1)
            if isinstance(cb, QComboBox):
                cb.setCurrentText(g)

        self._kw_default_seeded = True

    def del_kw_row(self):
        curr = self.kw_table.currentRow()
        if curr >= 0: self.kw_table.removeRow(curr)

    def load_keywords_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "키워드 파일 열기", "", "Excel/CSV (*.xlsx *.xls *.csv)")
        if not path:
            return
        try:
            if path.lower().endswith(".csv"):
                try:
                    df_kw = pd.read_csv(path, encoding="utf-8-sig")
                except Exception:
                    df_kw = pd.read_csv(path, encoding="cp949")
            else:
                df_kw = pd.read_excel(path)

            if df_kw is None or df_kw.empty:
                raise ValueError("키워드 파일이 비어 있습니다.")

            df_kw.columns = [str(c).strip() for c in df_kw.columns]

            kw_col = None
            gd_col = None
            for cand in ["키워드", "keyword", "Keyword", "KW"]:
                if cand in df_kw.columns:
                    kw_col = cand
                    break
            for cand in ["등급", "grade", "Grade", "단계"]:
                if cand in df_kw.columns:
                    gd_col = cand
                    break

            if kw_col is None or gd_col is None:
                if len(df_kw.columns) < 2:
                    raise ValueError("키워드 파일에는 최소 2개 컬럼(키워드/등급)이 필요합니다.")
                kw_col = df_kw.columns[0]
                gd_col = df_kw.columns[1]

            self.kw_table.setRowCount(0)
            grades_in_file = []

            for _, row in df_kw.iterrows():
                kw = "" if pd.isna(row[kw_col]) else str(row[kw_col]).strip()
                gd = "" if pd.isna(row[gd_col]) else str(row[gd_col]).strip()
                if not kw or not gd:
                    continue
                grades_in_file.append(gd)

                r = self.kw_table.rowCount()
                self.kw_table.insertRow(r)
                self.kw_table.setItem(r, 0, QTableWidgetItem(kw))

                cb = QComboBox()
                cb.setEditable(True)
                cb.addItems(self.get_grade_list())
                cb.setCurrentText(gd)
                self.kw_table.setCellWidget(r, 1, cb)

            # 업로드 파일 기준으로 등급목록
            if hasattr(self, "grade_order_edit") and self.grade_order_edit is not None:
                # 파일에서 등장한 등급만 추출
                unique_grades = []
                seen = set()
                for g in grades_in_file:
                    g = str(g).strip()
                    if not g or g in seen:
                        continue
                    seen.add(g)
                    unique_grades.append(g)

                # 정렬 숫자순, 그 외는 문자열/알파벳순
                def grade_sort_key(x: str):
                    x = str(x).strip()
                    m = re.match(r"^\s*(\d+)\s*등급\s*$", x)
                    if m:
                        return (0, int(m.group(1)))
                    if x.isdigit():
                        return (0, int(x))
                    if len(x) == 1 and x.isalpha():
                        return (1, ord(x.upper()) - ord("A"))
                    return (2, x)

                unique_grades = sorted(unique_grades, key=grade_sort_key)

                # 파일에 등급이 하나도 없으면 기본값 유지
                if not unique_grades:
                    unique_grades = ["A", "B", "C", "D"]

                # 등급목록을 파일 기준으로 교체
                self.grade_order_edit.setText(",".join(unique_grades))
                self.refresh_grade_comboboxes()

            QMessageBox.information(self, "완료", f"키워드 {self.kw_table.rowCount()}개를 불러왔습니다.")

        except Exception as e:
            QMessageBox.critical(self, "오류", f"키워드 파일을 불러오는 중 오류:\n{str(e)}")

    def show_mapping_dialog(self):
        diag = QDialog(self); diag.setWindowTitle("컬럼 매핑"); l = QVBoxLayout(diag); cols = list(self.current_df.columns); self.mappings = {}
        fields = [
            ("부서명 (대시보드용)", "dept_col"),
            ("문서명 (필수)", "text_col"),
            ("직위명", "extra_col")
        ]
        for label, key in fields:
            l.addWidget(QLabel(label)); cb = QComboBox(); cb.addItem("없음"); cb.addItems(cols)
            for i in range(cb.count()):
                if label.split()[0] in cb.itemText(i): cb.setCurrentIndex(i)
            l.addWidget(cb); self.mappings[key] = cb
        btn = QPushButton("분석 시작"); btn.clicked.connect(lambda: self.run_analysis(diag)); l.addWidget(btn); diag.exec()

    def run_analysis(self, diag):
        m_vals = {k: v.currentText() for k, v in self.mappings.items()}
        self.mapped_dept_col = m_vals.get('dept_col', '없음')
        kw_dict = {}
        KeywordOptions = None
        if self.selected_mode == "키워드 방식":
            Key_Opt = KeywordOptionsDialog(self)
            if Key_Opt.exec() != QDialog.Accepted:
                return
            KeywordOptions = Key_Opt.get_options()

            # 직위 기준 선택 시: 직위-등급 매핑(extra_dict) 입력 받기
            self.extra_dict = None
            if KeywordOptions == "extra":
                dlg = PositionMappingDialog(self)
                if dlg.exec() != QDialog.Accepted:
                    return
                self.extra_dict = dlg.get_mapping()

            for r in range(self.kw_table.rowCount()):
                k = self.kw_table.item(r, 0).text().strip()
                if k:
                    kw_dict[k] = self.kw_table.cellWidget(r, 1).currentText()

        ml_options = None
        if self.selected_mode == "머신러닝 방식":
            opt_dlg = MLOptionsDialog(self)
            if opt_dlg.exec() != QDialog.Accepted:
                return
            ml_options = opt_dlg.get_options()
            KeywordOptions = None

        diag.accept()
        self.pdia = QProgressDialog("데이터 분석 중...", "중단", 0, 100, self); self.pdia.show()
        self.thread = AnalysisThread(
            self.current_df, kw_dict, self.selected_mode, m_vals, 
            self.tokenizer, self.model, self.device, 
            keyword_mode=KeywordOptions, 
            ml_options=ml_options,
            extra_dict=self.extra_dict
        )
        self.thread.progress_signal.connect(self.pdia.setValue)
        self.thread.ask_ml_signal.connect(self.handle_ml_question)
        self.thread.finished_signal.connect(self.on_finished)
        self.thread.error_signal.connect(lambda msg: (self.pdia.close(), QMessageBox.critical(self, "분석 오류", msg)))
        self.thread.start()

    def handle_ml_question(self, count):
        rep = QMessageBox.question(self, "추가 분석", f"미분류 {count}건을 머신러닝으로 분석할까요?", QMessageBox.Yes|QMessageBox.No)
        self.thread.run_ml_after = (rep == QMessageBox.Yes)

    def on_finished(self, df):
        self.pdia.close()
        
        # Softmax값 확률로 변환
        for col in ['A', 'B', 'C', 'D']:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: f"{float(x)*100:.1f}%" if pd.notnull(x) else "")
        
        self.final_df = df
        
        self.display_df_on_table(self.preview_table, self.final_df)
        
        self.res_table.setRowCount(len(df))
        self.res_table.setColumnCount(len(df.columns))
        self.res_table.setHorizontalHeaderLabels(df.columns)
        
        for i in range(len(df)):
            for j in range(len(df.columns)):
                val = df.iloc[i, j]
                col_name = df.columns[j]
                item = QTableWidgetItem(str(val))
                
                if col_name == '최종 예측 결과':
                    item.setFont(QFont("Arial", 10, QFont.Bold))
                    item.setForeground(QColor("#2563EB") if "(머신러닝)" in str(val) else QColor("#2563EB"))
                
                self.res_table.setItem(i, j, item)

        # 버튼 상태 변경
        self.analyze_btn.setText("결과 엑셀 저장하기")
        self.analyze_btn.setStyleSheet("""
            QPushButton { background-color: #2563EB; color: white; font-weight: bold; border-radius: 8px; }
            QPushButton:hover { background-color: #234DAA; }
        """)
        
        try: self.analyze_btn.clicked.disconnect()
        except: pass
        self.analyze_btn.clicked.connect(self.save_excel)
        
        QMessageBox.information(self, "완료", "분석이 완료되었습니다.")

    def _collect_keywords_df(self) -> pd.DataFrame:
        """키워드 테이블(QTableWidget) → DataFrame"""
        if not hasattr(self, "kw_table") or self.kw_table is None:
            return pd.DataFrame(columns=["키워드", "등급"])

        rows = []
        for r in range(self.kw_table.rowCount()):
            k_item = self.kw_table.item(r, 0)
            keyword = "" if k_item is None else str(k_item.text()).strip()

            grade = ""
            cb = self.kw_table.cellWidget(r, 1)
            if isinstance(cb, QComboBox):
                grade = str(cb.currentText()).strip()

            if keyword and grade:
                rows.append({"키워드": keyword, "등급": grade})

        return pd.DataFrame(rows, columns=["키워드", "등급"])


    def _collect_position_mapping_df(self) -> pd.DataFrame:
        """직위 매핑(dict) → DataFrame"""
        mapping = getattr(self, "extra_dict", None)
        if not mapping:
            return pd.DataFrame(columns=["직위", "등급"])

        rows = [{"직위": str(k).strip(), "등급": str(v).strip()} for k, v in mapping.items() if str(k).strip() and str(v).strip()]
        return pd.DataFrame(rows, columns=["직위", "등급"])

    def export_keywords(self):
        """키워드 테이블을 엑셀(xlsx)로 내보냄"""
        kw_df = self._collect_keywords_df()

        if kw_df is None or kw_df.empty:
            QMessageBox.warning(self, "알림", "내보낼 키워드가 없습니다.\n키워드를 먼저 입력해 주세요.")
            return

        path, _ = QFileDialog.getSaveFileName(self, "키워드 내보내기", "키워드_내보내기.xlsx", "Excel (*.xlsx)")
        if not path:
            return
        if not path.lower().endswith(".xlsx"):
            path += ".xlsx"

        try:
            with pd.ExcelWriter(path, engine="openpyxl") as writer:
                kw_df.to_excel(writer, index=False, sheet_name="키워드")
            QMessageBox.information(self, "완료", "키워드가 엑셀로 저장되었습니다.")
        except Exception as e:
            QMessageBox.critical(self, "오류", f"키워드를 내보내는 중 오류가 발생했습니다:\n{str(e)}")


    def export_position_mapping(self):
        """직위-등급 매핑표(extra_dict)를 엑셀로 내보냄"""
        pos_df = self._collect_position_mapping_df()

        if pos_df is None or pos_df.empty:
            QMessageBox.warning(
                self, "알림",
                "내보낼 직위 매핑표가 없습니다.\n"
                "키워드 방식 → 중복 키워드 처리 옵션에서 '직위 기준'을 선택하고\n"
                "직위-등급 매핑표를 입력/적용한 뒤 다시 시도해 주세요."
            )
            return

        path, _ = QFileDialog.getSaveFileName(self, "직위 매핑 내보내기", "직위매핑_내보내기.xlsx", "Excel (*.xlsx)")
        if not path:
            return
        if not path.lower().endswith(".xlsx"):
            path += ".xlsx"

        try:
            with pd.ExcelWriter(path, engine="openpyxl") as writer:
                pos_df.to_excel(writer, index=False, sheet_name="직위매핑")
            QMessageBox.information(self, "완료", "직위 매핑표가 엑셀로 저장되었습니다.")
        except Exception as e:
            QMessageBox.critical(self, "오류", f"직위 매핑표를 내보내는 중 오류가 발생했습니다:\n{str(e)}")


    def save_excel(self):
        if self.final_df is None: return
        
        path, _ = QFileDialog.getSaveFileName(self, "저장", "분류결과.xlsx", "Excel (*.xlsx)")
        if path:
            self.final_df.to_excel(path, index=False)
            QMessageBox.information(self, "완료", "엑셀 파일이 저장되었습니다.")

if __name__ == "__main__":  
    myappid = 'LeeWonjung-KwonSeulhee.Document_classification_system.v1' 
    #ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
    app = QApplication(sys.argv)
    icon = QIcon(resource_path("assets/icon.ico"))
    app.setWindowIcon(icon) 
    w = MainWindow()
    w.setWindowIcon(icon)
    w.show()
    sys.exit(app.exec())