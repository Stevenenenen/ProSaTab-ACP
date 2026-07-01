# app_tkinter.py
# -*- coding: utf-8 -*-

import threading
import traceback
import tkinter as tk
from tkinter import messagebox

from predictor_core import ACPPredictor


class ACPApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ACP Predictor")
        self.root.geometry("980x650")
        self.root.minsize(900, 620)
        self.root.resizable(True, True)

        self.predictor = None
        self.is_predicting = False

        self.bg_color = "#F4F6F8"
        self.card_color = "#FFFFFF"
        self.primary_color = "#2563EB"
        self.primary_dark = "#1D4ED8"
        self.text_color = "#111827"
        self.sub_text_color = "#6B7280"
        self.success_color = "#15803D"
        self.warning_color = "#B45309"
        self.error_color = "#DC2626"
        self.border_color = "#D1D5DB"

        self.root.configure(bg=self.bg_color)

        self.build_ui()
        self.load_predictor_async()

    def build_ui(self):
        main_frame = tk.Frame(self.root, bg=self.bg_color)
        main_frame.pack(fill="both", expand=True, padx=36, pady=24)

        title = tk.Label(
            main_frame,
            text="Anticancer Peptide Predictor",
            font=("Arial", 24, "bold"),
            bg=self.bg_color,
            fg=self.text_color,
        )
        title.pack(pady=(0, 8))

        subtitle = tk.Label(
            main_frame,
            text="Input an amino acid sequence to predict ACP / non-ACP",
            font=("Arial", 11),
            bg=self.bg_color,
            fg=self.sub_text_color,
        )
        subtitle.pack(pady=(0, 10))

        self.status_label = tk.Label(
            main_frame,
            text="Loading model, please wait...",
            font=("Arial", 11),
            bg=self.bg_color,
            fg=self.warning_color,
        )
        self.status_label.pack(pady=(0, 16))

        input_card = tk.Frame(
            main_frame,
            bg=self.card_color,
            highlightbackground=self.border_color,
            highlightthickness=1,
        )
        input_card.pack(fill="x", pady=(0, 16))

        input_header = tk.Frame(input_card, bg=self.card_color)
        input_header.pack(fill="x", padx=18, pady=(14, 6))

        input_label = tk.Label(
            input_header,
            text="Input sequence",
            font=("Arial", 12, "bold"),
            bg=self.card_color,
            fg=self.text_color,
        )
        input_label.pack(side="left")

        self.length_label = tk.Label(
            input_header,
            text="Length: 0",
            font=("Arial", 10),
            bg=self.card_color,
            fg=self.sub_text_color,
        )
        self.length_label.pack(side="right")

        self.sequence_text = tk.Text(
            input_card,
            width=88,
            height=6,
            font=("Consolas", 12),
            wrap="word",
            relief="flat",
            bg="#FAFAFA",
            fg=self.text_color,
            insertbackground=self.text_color,
            padx=10,
            pady=10,
        )
        self.sequence_text.pack(padx=18, pady=(0, 14), fill="x")
        self.sequence_text.bind("<KeyRelease>", self.update_sequence_length)

        button_frame = tk.Frame(main_frame, bg=self.bg_color)
        button_frame.pack(pady=(0, 18))

        self.predict_button = tk.Button(
            button_frame,
            text="Predict",
            width=18,
            height=2,
            font=("Arial", 11, "bold"),
            bg=self.primary_color,
            fg="white",
            activebackground=self.primary_dark,
            activeforeground="white",
            relief="flat",
            cursor="hand2",
            command=self.predict_async,
            state=tk.DISABLED,
        )
        self.predict_button.grid(row=0, column=0, padx=12)

        self.clear_button = tk.Button(
            button_frame,
            text="Clear",
            width=18,
            height=2,
            font=("Arial", 11),
            bg="#E5E7EB",
            fg=self.text_color,
            activebackground="#D1D5DB",
            activeforeground=self.text_color,
            relief="flat",
            cursor="hand2",
            command=self.clear,
        )
        self.clear_button.grid(row=0, column=1, padx=12)

        result_card = tk.Frame(
            main_frame,
            bg=self.card_color,
            highlightbackground=self.border_color,
            highlightthickness=1,
        )
        result_card.pack(fill="x", pady=(0, 14))

        result_title = tk.Label(
            result_card,
            text="Prediction result",
            font=("Arial", 12, "bold"),
            bg=self.card_color,
            fg=self.text_color,
        )
        result_title.pack(anchor="w", padx=18, pady=(14, 8))

        result_grid = tk.Frame(result_card, bg=self.card_color)
        result_grid.pack(fill="x", padx=18, pady=(0, 18))

        self.prediction_value = self.create_result_row(
            result_grid,
            row=0,
            label="Prediction",
            value="-",
        )

        self.probability_value = self.create_result_row(
            result_grid,
            row=1,
            label="Probability ACP",
            value="-",
        )

    def create_result_row(self, parent, row, label, value):
        label_widget = tk.Label(
            parent,
            text=label + ":",
            font=("Arial", 11),
            bg=self.card_color,
            fg=self.sub_text_color,
            width=22,
            anchor="w",
        )
        label_widget.grid(row=row, column=0, sticky="w", pady=8)

        value_widget = tk.Label(
            parent,
            text=value,
            font=("Arial", 15, "bold"),
            bg=self.card_color,
            fg=self.text_color,
            anchor="w",
        )
        value_widget.grid(row=row, column=1, sticky="w", pady=8)

        return value_widget

    def update_sequence_length(self, event=None):
        sequence = self.sequence_text.get("1.0", tk.END).strip()
        sequence = "".join(sequence.split())
        self.length_label.config(text=f"Length: {len(sequence)}")

    def set_status(self, text, color=None):
        if color is None:
            color = self.sub_text_color
        self.status_label.config(text=text, fg=color)

    def load_predictor_async(self):
        thread = threading.Thread(target=self.load_predictor, daemon=True)
        thread.start()

    def load_predictor(self):
        try:
            self.root.after(
                0,
                lambda: self.set_status(
                    "Loading feature extractors, selector chain and final model...",
                    self.warning_color,
                ),
            )

            self.predictor = ACPPredictor(
                model_dir="models",
                feature_device="cpu",
                tabpfn_device="cuda",
            )

            self.root.after(0, self.on_model_loaded)

        except Exception:
            err_msg = traceback.format_exc()
            self.root.after(0, lambda msg=err_msg: self.on_model_error(msg))

    def on_model_loaded(self):
        self.set_status("Model loaded successfully. Ready to predict.", self.success_color)
        self.predict_button.config(state=tk.NORMAL)

    def on_model_error(self, error_msg):
        self.set_status("Model loading failed. Please check the console output.", self.error_color)

        print("\n[MODEL LOADING ERROR]")
        print(error_msg)

        messagebox.showerror("Model Error", error_msg)

    def predict_async(self):
        sequence = self.sequence_text.get("1.0", tk.END).strip()

        if not sequence:
            messagebox.showwarning("Input Error", "Please input an amino acid sequence.")
            return

        if self.predictor is None:
            messagebox.showwarning("Model Error", "Model is not loaded yet.")
            return

        if self.is_predicting:
            return

        self.is_predicting = True
        self.predict_button.config(state=tk.DISABLED)
        self.clear_button.config(state=tk.DISABLED)

        self.set_status(
            "Predicting... Feature extraction and TabPFN inference may take some time.",
            self.warning_color,
        )

        self.reset_result_for_running()

        thread = threading.Thread(
            target=self.run_prediction,
            args=(sequence,),
            daemon=True,
        )
        thread.start()

    def reset_result_for_running(self):
        self.prediction_value.config(text="Running...", fg=self.warning_color)
        self.probability_value.config(text="Running...", fg=self.warning_color)

    def run_prediction(self, sequence):
        try:
            result = self.predictor.predict(sequence)
            self.root.after(0, lambda res=result: self.show_result(res))

        except Exception:
            err_msg = traceback.format_exc()
            self.root.after(0, lambda msg=err_msg: self.show_error(msg))

    def show_result(self, result):
        prediction = result.get("prediction", "-")
        prob = result.get("probability_acp", None)

        if prediction == "ACP":
            pred_color = self.success_color
        else:
            pred_color = self.warning_color

        self.prediction_value.config(
            text=str(prediction),
            fg=pred_color,
        )

        if prob is not None:
            self.probability_value.config(
                text=f"{float(prob):.4f}",
                fg=self.text_color,
            )
        else:
            self.probability_value.config(
                text="-",
                fg=self.text_color,
            )

        self.set_status(
            "Prediction completed. The result is shown below.",
            self.success_color,
        )

        self.predict_button.config(state=tk.NORMAL)
        self.clear_button.config(state=tk.NORMAL)
        self.is_predicting = False

        self.root.update_idletasks()

    def show_error(self, error_msg):
        self.set_status("Prediction failed. Please check the console output.", self.error_color)

        self.prediction_value.config(text="Failed", fg=self.error_color)
        self.probability_value.config(text="-", fg=self.text_color)

        self.predict_button.config(state=tk.NORMAL)
        self.clear_button.config(state=tk.NORMAL)
        self.is_predicting = False

        print("\n[PREDICTION ERROR]")
        print(error_msg)

        messagebox.showerror("Prediction Error", error_msg)

    def clear(self):
        if self.is_predicting:
            return

        self.sequence_text.delete("1.0", tk.END)
        self.length_label.config(text="Length: 0")

        self.prediction_value.config(text="-", fg=self.text_color)
        self.probability_value.config(text="-", fg=self.text_color)

        if self.predictor is not None:
            self.set_status("Model loaded successfully. Ready to predict.", self.success_color)
            self.predict_button.config(state=tk.NORMAL)
        else:
            self.set_status("Loading model, please wait...", self.warning_color)
            self.predict_button.config(state=tk.DISABLED)

        self.clear_button.config(state=tk.NORMAL)
        self.root.update_idletasks()


def main():
    root = tk.Tk()
    ACPApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()