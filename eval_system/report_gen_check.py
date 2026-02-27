import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class ConfusionMetrics:
    letter: str
    worksheet_num: int
    is_capital: bool
    true_positives: int  = 0  # AI says incorrect AND voter says incorrect
    true_negatives: int  = 0  # AI says correct   AND voter says correct
    false_positives: int = 0  # AI says incorrect BUT voter says correct
    false_negatives: int = 0  # AI says correct   BUT voter says incorrect

    def accuracy(self) -> float:
        total = self.true_positives + self.true_negatives + self.false_positives + self.false_negatives
        return (self.true_positives + self.true_negatives) / total if total > 0 else 0.0

    def precision(self) -> float:
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom > 0 else 0.0

    def recall(self) -> float:
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom > 0 else 0.0

    def f1_score(self) -> float:
        p, r = self.precision(), self.recall()
        return 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0


class WorksheetValidator:
    def __init__(self, excel_path: str):
        self.excel_path = excel_path
        self.home_df    = pd.read_excel(excel_path, sheet_name='Home')
        self.home_df.columns = self.home_df.columns.str.strip()

        self.cap_rep_cols = ['Unnamed: 1', 'Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4', 'Unnamed: 5']
        self.sml_rep_cols = ['Unnamed: 10', 'Unnamed: 11', 'Unnamed: 12', 'Unnamed: 13', 'Unnamed: 14']
        self.voters       = ['Jash', 'Saimon', 'Ysa', 'Sean']

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _load_sheet(self, worksheet_num: int) -> pd.DataFrame:
        """Load and normalize a worksheet sheet."""
        df = pd.read_excel(self.excel_path, sheet_name=f'W{worksheet_num}')
        df.columns = df.columns.str.strip()
        # Normalize first column - W1 uses 'Capital', W2+ uses 'Capital Letter'
        df = df.rename(columns={df.columns[0]: 'Capital'})
        return df

    def _extract_letter(
        self,
        letter: str,
        is_capital: bool,
        num_incorrect_val,
        given_grade_val,
        voter_rows: list,
        rep_cols: list,
        letter_position: int
    ) -> Optional[Dict]:
        """
        Return a dict for ALL letters.
        Correct letters get empty incorrect_indices [].
        Incorrect letters get the list of repetition indices voted incorrect by majority.
        """
        if not pd.notna(num_incorrect_val):
            num_inc = 0
        else:
            val_str = str(num_incorrect_val).strip()
            if val_str in ['', 'no grade']:
                num_inc = 0
            else:
                try:
                    num_inc = int(float(val_str))
                except (ValueError, TypeError):
                    num_inc = 0

        num_voters = len(voter_rows)
        majority   = (num_voters // 2) + 1  # e.g. 3 voters -> 2, 4 voters -> 3

        incorrect_indices = []
        for i, col in enumerate(rep_cols, start=1):
            votes = [r[col] for r in voter_rows if pd.notna(r[col])]
            if sum(v == 1.0 for v in votes) >= majority:
                incorrect_indices.append(i)

        return {
            'letter':            letter,
            'is_capital':        is_capital,
            'position':          letter_position,
            'num_incorrect':     num_inc,
            'given_grade':       given_grade_val,
            'incorrect_indices': incorrect_indices,  # [] means all correct
        }

    def _parse_worksheet(self, worksheet_num: int) -> List[Dict]:
        """
        Parse one worksheet sheet and return ALL letters,
        including correctly written ones (incorrect_indices = []).
        """
        df             = self._load_sheet(worksheet_num)
        results        = []
        current_cap    = None
        current_sml    = None
        letter_pos     = 0
        letter_row_idx = None
        voter_rows     = []

        def flush():
            if not current_cap or not voter_rows:
                return
            prev = df.loc[letter_row_idx]
            cap  = self._extract_letter(
                current_cap, True,
                prev['No. of Incorrect'], prev['Given Grade'],
                voter_rows, self.cap_rep_cols, letter_pos
            )
            sml  = self._extract_letter(
                current_sml, False,
                prev['# of Incorrect'], prev['Given Grade.1'],
                voter_rows, self.sml_rep_cols, letter_pos
            )
            if cap: results.append(cap)
            if sml: results.append(sml)

        for idx, row in df.iterrows():
            capital = str(row['Capital']).strip()
            small   = str(row['Small Letters']).strip()

            if capital in list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'):
                flush()
                current_cap    = capital
                current_sml    = small if small in list('abcdefghijklmnopqrstuvwxyz') else capital.lower()
                letter_pos    += 1
                letter_row_idx = idx
                voter_rows     = []
            elif capital in self.voters:
                voter_rows.append(row)

        flush()  # process last letter block
        return results

    # ------------------------------------------------------------------ #
    #  Public: print voting results                                        #
    # ------------------------------------------------------------------ #

    def print_voting_results(self, worksheet_num: int):
        """Print all majority-voted incorrect letters for a worksheet."""
        results   = self._parse_worksheet(worksheet_num)
        incorrect = [r for r in results if r['incorrect_indices']]

        print(f"\n{'='*55}")
        print(f"Worksheet {worksheet_num} - Voting Results")
        print(f"{'='*55}")

        if not incorrect:
            print("  No letters marked incorrect by majority vote.")
        else:
            for r in incorrect:
                case_label = "Capital" if r['is_capital'] else "Small"
                print(f"  [{case_label}] Letter    : {r['letter']}")
                print(f"  Position          : {r['position']}")
                print(f"  # Incorrect       : {r['num_incorrect']}/5 repetitions")
                print(f"  Given Grade       : {r['given_grade']}")
                print(f"  Incorrect Indices : {r['incorrect_indices']}")
                print(f"  {'-'*45}")
        print()

    def print_all_voting_results(self):
        """Print voting results for all worksheets."""
        for ws_num in self.home_df['Worksheet'].dropna().astype(int):
            self.print_voting_results(ws_num)

    # ------------------------------------------------------------------ #
    #  Public: ground truth                                                #
    # ------------------------------------------------------------------ #

    def get_ground_truth(self, worksheet_num: int) -> Dict[str, List[int]]:
        """
        Return ground truth for ALL letters as:
            { 'A': [1, 3], 'b': [2], 'C': [] }
        Empty list means all repetitions were correct.
        """
        results = self._parse_worksheet(worksheet_num)
        return {r['letter']: r['incorrect_indices'] for r in results}

    # ------------------------------------------------------------------ #
    #  Public: validation / confusion matrix                              #
    # ------------------------------------------------------------------ #

    def calculate_confusion_matrix(
        self,
        worksheet_num: int,
        letter_instances: List[Dict],
        threshold_score: float = 75.0
    ) -> List[ConfusionMetrics]:
        """
        Cross-reference AI letter_instances against voter ground truth.
        Operates at repetition level (each of 5 reps is a TP/TN/FP/FN).
        A repetition is AI-predicted incorrect if avg score < threshold_score.

        Only letters present in letter_instances are evaluated -
        letters with no AI score are excluded rather than counted as FN.
        """
        ground_truth = self.get_ground_truth(worksheet_num)

        # Build AI predictions and track which letters were actually scored.
        # ai_predictions : { 'A': [1, 3] }  - only reps below threshold
        # ai_scored_letters: { 'A', 'B', ... } - all letters AI produced a score for
        ai_predictions: Dict[str, List[int]] = {}
        ai_scored_letters: set = set()

        for inst in letter_instances:
            letter  = inst['letter']
            rep_num = inst['repetition_num']
            avg     = (
                inst['letter_form'] +
                inst['size'] +
                inst['line_align']
            ) / 3

            ai_scored_letters.add(letter)  # mark as scored regardless of result

            if avg < threshold_score:
                ai_predictions.setdefault(letter, []).append(rep_num)

        # Only evaluate letters the AI actually scored -
        # avoids false FNs for letters with no AI data
        all_letters = set(ground_truth.keys()) & ai_scored_letters
        all_reps    = list(range(1, 6))  # repetitions 1-5

        metrics_list = []
        for letter in sorted(all_letters):
            gt_incorrect = set(ground_truth.get(letter, []))
            ai_incorrect = set(ai_predictions.get(letter, []))
            is_capital   = letter.isupper()

            metric = ConfusionMetrics(
                letter=letter,
                worksheet_num=worksheet_num,
                is_capital=is_capital
            )

            for rep in all_reps:
                voter_says = rep in gt_incorrect
                ai_says    = rep in ai_incorrect

                if     ai_says and     voter_says:   metric.true_positives  += 1
                elif not ai_says and not voter_says:  metric.true_negatives  += 1
                elif   ai_says and not voter_says:   metric.false_positives += 1
                elif not ai_says and     voter_says:  metric.false_negatives += 1

            metrics_list.append(metric)

        return metrics_list

    def generate_validation_report(
        self,
        all_worksheet_data: Dict[int, List[Dict]],
        threshold_score: float = 75.0
    ) -> pd.DataFrame:
        """Generate full TP/TN/FP/FN report across all worksheets."""
        rows = []
        for ws_num, letter_instances in all_worksheet_data.items():
            # Pre-compute rep counts per letter from letter_instances
            # Count how many times each letter was recognized by the AI
            rep_count: Dict[str, int] = {}
            for inst in letter_instances:
                letter = inst['letter']
                rep_count[letter] = rep_count.get(letter, 0) + 1

            for m in self.calculate_confusion_matrix(ws_num, letter_instances, threshold_score):
                rows.append({
                    'Worksheet':         m.worksheet_num,
                    'Letter':            m.letter,
                    'Case':              'Capital' if m.is_capital else 'Small',
                    'Repetitions Found': rep_count.get(m.letter, 0),
                    'TP':                m.true_positives,
                    'TN':                m.true_negatives,
                    'FP':                m.false_positives,
                    'FN':                m.false_negatives,
                    'Accuracy':          round(m.accuracy(),  4),
                    'Precision':         round(m.precision(), 4),
                    'Recall':            round(m.recall(),    4),
                    'F1':                round(m.f1_score(),  4),
                })
        return pd.DataFrame(rows)

    def print_validation_report(
        self,
        all_worksheet_data: Dict[int, List[Dict]],
        threshold_score: float = 75.0
    ):
        """Print voting results alongside TP/TN/FP/FN confusion matrix."""
        for ws_num, letter_instances in all_worksheet_data.items():
            self.print_voting_results(ws_num)

            metrics = self.calculate_confusion_matrix(ws_num, letter_instances, threshold_score)
            if not metrics:
                continue

            print(f"  {'Letter':<8} {'Case':<8} {'TP':>4} {'TN':>4} {'FP':>4} {'FN':>4} "
                  f"{'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7}")
            print(f"  {'-'*65}")
            for m in metrics:
                case = 'Capital' if m.is_capital else 'Small'
                print(
                    f"  {m.letter:<8} {case:<8} "
                    f"{m.true_positives:>4} {m.true_negatives:>4} "
                    f"{m.false_positives:>4} {m.false_negatives:>4} "
                    f"{m.accuracy():>7.2%} {m.precision():>7.2%} "
                    f"{m.recall():>7.2%} {m.f1_score():>7.2%}"
                )
            print()

    def generate_summary_statistics(self, report_df: pd.DataFrame) -> Dict:
        """Overall summary statistics across all worksheets."""
        tp    = report_df['TP'].sum()
        tn    = report_df['TN'].sum()
        fp    = report_df['FP'].sum()
        fn    = report_df['FN'].sum()
        total = tp + tn + fp + fn

        return {
            'total_predictions': int(total),
            'true_positives':    int(tp),
            'true_negatives':    int(tn),
            'false_positives':   int(fp),
            'false_negatives':   int(fn),
            'overall_accuracy':  round((tp + tn) / total, 4) if total else 0,
            'overall_precision': round(tp / (tp + fp),    4) if (tp + fp) else 0,
            'overall_recall':    round(tp / (tp + fn),    4) if (tp + fn) else 0,
        }


    def generate_letter_averages(
        self,
        letter_instances: List[Dict],
        grade_thresholds: Dict[str, int] = None
    ) -> pd.DataFrame:
        """
        Generate per-letter average scores across all repetitions.
        Returns a DataFrame with Avg Form, Avg Size, Avg Align, Overall Avg,
        Grade, Best rep score, and Worst rep score.
        """
        if grade_thresholds is None:
            grade_thresholds = {'A': 95, 'B': 85, 'C': 75, 'D': 60}

        def score_to_grade(score: float) -> str:
            if score >= grade_thresholds['A']: return 'A'
            if score >= grade_thresholds['B']: return 'B'
            if score >= grade_thresholds['C']: return 'C'
            return 'D'

        # Group instances by letter, deduplicating by (letter, repetition_num)
        # so duplicate entries from multiple append_report calls don't inflate counts
        grouped: Dict[str, list] = {}
        seen: set = set()
        for inst in letter_instances:
            key = (inst['letter'], inst['repetition_num'])
            if key not in seen:
                seen.add(key)
                grouped.setdefault(inst['letter'], []).append(inst)

        rows = []
        for letter in sorted(grouped.keys()):
            reps  = grouped[letter]
            forms = [r['letter_form'] for r in reps]
            sizes = [r['size']        for r in reps]
            aligns= [r['line_align']  for r in reps]
            avgs  = [round((f + s + a) / 3, 1) for f, s, a in zip(forms, sizes, aligns)]

            avg_form    = round(sum(forms)  / len(forms),  1)
            avg_size    = round(sum(sizes)  / len(sizes),  1)
            avg_align   = round(sum(aligns) / len(aligns), 1)
            overall_avg = round(sum(avgs)   / len(avgs),   1)

            rows.append({
                'Letter':      letter,
                # 'Case':        'Capital' if letter.isupper() else 'Small',
                'Avg Form':    avg_form,
                'Avg Size':    avg_size,
                'Avg Align':   avg_align,
                'Overall Avg': overall_avg,
                'Grade':       score_to_grade(overall_avg),
                'Best':        round(max(avgs), 1),
                'Worst':       round(min(avgs), 1),
                'Rep Count':   len(reps),
            })

        return pd.DataFrame(rows)

    def generate_individual_repetitions(
        self,
        letter_instances: List[Dict]
    ) -> pd.DataFrame:
        """
        Generate a flat table of all individual repetitions with per-rep avg score.
        """
        rows = []
        seen: set = set()
        for inst in letter_instances:
            key = (inst['letter'], inst['repetition_num'])
            if key in seen:
                continue
            seen.add(key)
            avg = round((inst['letter_form'] + inst['size'] + inst['line_align']) / 3, 1)
            rows.append({
                'Letter':      inst['letter'],
                # 'Case':        'Capital' if inst['letter'].isupper() else 'Small',
                'Rep':         inst['repetition_num'],
                'Form':        inst['letter_form'],
                'Size':        inst['size'],
                'Align':       inst['line_align'],
                'Avg':         avg,
            })

        return pd.DataFrame(rows)

    def export_to_excel(
        self,
        report_df: pd.DataFrame,
        letter_instances: List[Dict] = None,
        output_path: str = 'validation_report.xlsx',
        grade_thresholds: Dict[str, int] = None
    ):
        """
        Export to Excel with up to four sheets:
            - 'Validation Report'       : full TP/TN/FP/FN per letter per worksheet
            - 'Summary'                 : overall aggregated statistics
            - 'Letter Averages'         : per-letter avg scores, grade, best/worst (if letter_instances provided)
            - 'Individual Repetitions'  : all individual rep scores (if letter_instances provided)
        """
        summary    = self.generate_summary_statistics(report_df)
        summary_df = pd.DataFrame([summary])

        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Sheet 1: full validation report
            report_df.to_excel(writer, sheet_name='Validation Report', index=False)

            # Sheet 2: summary statistics
            summary_df.to_excel(writer, sheet_name='Summary', index=False)

            # Sheet 3 & 4: only if letter_instances provided
            if letter_instances:
                letter_avg_df = self.generate_letter_averages(letter_instances, grade_thresholds)
                letter_avg_df.to_excel(writer, sheet_name='Letter Averages', index=False)

                rep_df = self.generate_individual_repetitions(letter_instances)
                rep_df.to_excel(writer, sheet_name='Individual Repetitions', index=False)

            # Auto-fit column widths for all sheets
            for sheet_name in writer.sheets:
                ws = writer.sheets[sheet_name]
                for col in ws.columns:
                    max_len = max(
                        len(str(cell.value)) if cell.value is not None else 0
                        for cell in col
                    )
                    ws.column_dimensions[col[0].column_letter].width = max_len + 4

        print(f"Report exported to {output_path}")


# ------------------------------------------------------------------ #
#  Usage example                                                       #
# ------------------------------------------------------------------ #
if __name__ == '__main__':
    validator = WorksheetValidator('Voting_Page___1_.xlsx')

    # Print voting results only
    validator.print_voting_results(2)

    # Example letter_instances from AI grading
    letter_instances = [
        {'letter': 'C', 'repetition_num': 2, 'letter_form': 60.0, 'size': 62.0, 'line_align': 58.0},
        {'letter': 'C', 'repetition_num': 4, 'letter_form': 55.0, 'size': 57.0, 'line_align': 53.0},
        {'letter': 'K', 'repetition_num': 1, 'letter_form': 88.0, 'size': 90.0, 'line_align': 85.0},
        {'letter': 'k', 'repetition_num': 1, 'letter_form': 60.0, 'size': 62.0, 'line_align': 58.0},
        {'letter': 'k', 'repetition_num': 2, 'letter_form': 70.0, 'size': 72.0, 'line_align': 68.0},
    ]

    # Print voting + confusion matrix together
    validator.print_validation_report({2: letter_instances})

    # Generate DataFrame report
    report_df = validator.generate_validation_report({2: letter_instances})
    print(report_df.to_string(index=False))

    # Summary statistics
    summary = validator.generate_summary_statistics(report_df)
    print("\nSummary Statistics:")
    for key, val in summary.items():
        print(f"  {key}: {val}")
