from cv2 import line
import pandas as pd
from typing import List, Dict, Tuple
from dataclasses import dataclass

@dataclass
class ConfusionMetrics:
    """Store confusion matrix metrics for a letter"""
    letter: str
    worksheet_num: int
    true_positives: int = 0   # AI says incorrect AND voter says incorrect
    true_negatives: int = 0   # AI says correct AND voter says correct
    false_positives: int = 0  # AI says incorrect BUT voter says correct
    false_negatives: int = 0  # AI says correct BUT voter says incorrect
    
    def accuracy(self) -> float:
        total = self.true_positives + self.true_negatives + self.false_positives + self.false_negatives
        return (self.true_positives + self.true_negatives) / total if total > 0 else 0.0
    
    def precision(self) -> float:
        denominator = self.true_positives + self.false_positives
        return self.true_positives / denominator if denominator > 0 else 0.0
    
    def recall(self) -> float:
        denominator = self.true_positives + self.false_negatives
        return self.true_positives / denominator if denominator > 0 else 0.0
    
    def f1_score(self) -> float:
        p = self.precision()
        r = self.recall()
        return 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0


class WorksheetValidator:
    def __init__(self, excel_path: str, threshold: int = 2):
        """
        Initialize validator
        
        Args:
            excel_path: Path to the voting Excel file
            threshold: Number of TRUE votes needed to mark letter as incorrect (default: 3)
        """
        self.excel_path = excel_path
        self.threshold = threshold
        self.voting_data = self._load_voting_data()
    
    def _load_voting_data(self) -> pd.DataFrame:
        """Load the Home sheet from Excel"""
        df = pd.read_excel(self.excel_path, sheet_name='Home')
        return df
    
    def get_incorrect_letters_from_votes(self, worksheet_num: int) -> List[str]:
        """
        Parse the worksheet sheet to find letters marked incorrect by voters
        
        Args:
            worksheet_num: Worksheet number to check
            
        Returns:
            List of letters (e.g., ['A', 'C', 'M']) that voters marked as incorrect
        """
        sheet_name = f'W{worksheet_num}'
        
        try:
            df = pd.read_excel(self.excel_path, sheet_name=sheet_name)
            df.columns = df.columns.str.strip()                          # strip whitespace from all column names
            df = df.rename(columns={df.columns[0]: 'Capital'})           # normalize 'Capital Letter' → 'Capital'
            
            incorrect_letters = []
            
            # Iterate through each row (each letter)
            for idx, row in df.iterrows():
                # Count TRUE values in the row (excluding letter column)
                # Adjust column names based on your actual Excel structure
                true_count = 0
                
                for col in df.columns:
                    if col.lower() not in ['letter', 'worksheet', 'unnamed']:
                        if pd.notna(row[col]) and str(row[col]).upper() == 'TRUE':
                            true_count += 1
                
                # If threshold met, mark letter as incorrect
                if true_count >= self.threshold:
                    letter = row.get('Letter', row.get('letter', None))
                    if letter and pd.notna(letter):
                        incorrect_letters.append(str(letter).upper())
            
            return incorrect_letters
            
        except Exception as e:
            print(f"Error reading worksheet {worksheet_num}: {e}")
            return []
    
    def predict_incorrect_letters_from_ai(self, letter_instances: List[Dict], 
                                          threshold_score: float = 75.0) -> List[str]:
        """
        Determine which letters AI thinks are incorrect based on scores
        
        Args:
            letter_instances: List of letter instance dictionaries
            threshold_score: Score below which letter is considered incorrect
            
        Returns:
            List of letters AI predicts as incorrect
        """
        incorrect_letters = []
        
        for instance in letter_instances:
            letter = instance['letter']
            
            # Calculate average score for this letter instance
            avg_score = ( instance['letter_form'] + instance['size'] + instance['line_align'] ) / 3
            
            if avg_score < threshold_score:
                incorrect_letters.append(letter)
        
        return list(set(incorrect_letters))  # Remove duplicates
    
    def calculate_confusion_matrix(self, worksheet_num: int, 
                                   letter_instances: List[Dict],
                                   grade_lvl: str = "grade1",
                                   ) -> List[ConfusionMetrics]:
        """
        Calculate confusion matrix for all letters in a worksheet
        
        Args:
            worksheet_num: Worksheet number
            letter_instances: AI letter predictions
            threshold_score: Score threshold for AI predictions
            
        Returns:
            List of ConfusionMetrics for each unique letter
        """
        thresholds = {}

        grade_thresh = {
            'kinder':(40, 60, 50),
            'grade1':(60, 80, 70),
            'grade2':(80, 90, 95),
        }
        form, line_align, size = grade_thresh[grade_lvl]
        threshold_score = round((form + line_align + size) / 3, 2)
        print(f"threshold_score: {threshold_score}")
        # Get ground truth from voters
        voter_incorrect = set(self.get_incorrect_letters_from_votes(worksheet_num))
        
        # Get AI predictions
        ai_incorrect = set(self.predict_incorrect_letters_from_ai(letter_instances, threshold_score))
        
        # Get all unique letters in the worksheet
        all_letters = set(instance['letter'] for instance in letter_instances)
        
        metrics_list = []
        
        for letter in sorted(all_letters):
            metric = ConfusionMetrics(letter=letter, worksheet_num=worksheet_num)
            
            voter_says_incorrect = letter in voter_incorrect
            ai_says_incorrect = letter in ai_incorrect
            
            if ai_says_incorrect and voter_says_incorrect:
                metric.true_positives = 1
            elif not ai_says_incorrect and not voter_says_incorrect:
                metric.true_negatives = 1
            elif ai_says_incorrect and not voter_says_incorrect:
                metric.false_positives = 1
            elif not ai_says_incorrect and voter_says_incorrect:
                metric.false_negatives = 1
            
            metrics_list.append(metric)
        
        return metrics_list
    
    def generate_validation_report(self, all_worksheet_data: Dict[int, List[Dict]],
                                   threshold_score: float = 75.0) -> pd.DataFrame:
        """
        Generate comprehensive validation report across all worksheets
        
        Args:
            all_worksheet_data: Dict mapping worksheet_num to letter_instances
            threshold_score: Score threshold for AI predictions
            
        Returns:
            DataFrame with validation results
        """
        all_metrics = []
        
        for worksheet_num, letter_instances in all_worksheet_data.items():
            metrics = self.calculate_confusion_matrix(
                worksheet_num, 
                letter_instances, 
            )
            all_metrics.extend(metrics)
        
        # Convert to DataFrame
        data = []
        for metric in all_metrics:
            data.append({
                'Worksheet': metric.worksheet_num,
                'Letter': metric.letter,
                'TP': metric.true_positives,
                'TN': metric.true_negatives,
                'FP': metric.false_positives,
                'FN': metric.false_negatives,
                'Accuracy': metric.accuracy(),
                'Precision': metric.precision(),
                'Recall': metric.recall(),
                'F1_Score': metric.f1_score()
            })
        
        df = pd.DataFrame(data)
        return df
    
    def generate_summary_statistics(self, report_df: pd.DataFrame) -> Dict:
        """Generate overall summary statistics"""
        total_tp = report_df['TP'].sum()
        total_tn = report_df['TN'].sum()
        total_fp = report_df['FP'].sum()
        total_fn = report_df['FN'].sum()
        
        total = total_tp + total_tn + total_fp + total_fn
        
        return {
            'total_predictions': total,
            'true_positives': total_tp,
            'true_negatives': total_tn,
            'false_positives': total_fp,
            'false_negatives': total_fn,
            'overall_accuracy': (total_tp + total_tn) / total if total > 0 else 0,
            'overall_precision': total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0,
            'overall_recall': total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0,
            'agreement_rate': (total_tp + total_tn) / total if total > 0 else 0
        }


# Usage Example
    def print_voting_results(self, worksheet_num: int):
        sheet_name = f'W{worksheet_num}'

        try:
            df = pd.read_excel(self.excel_path, sheet_name=sheet_name)
            df.columns = df.columns.str.strip()

            # Normalize first column name — W1 uses 'Capital', W2+ uses 'Capital Letter'
            first_col = df.columns[0]
            if first_col in ('Capital', 'Capital Letter'):
                df = df.rename(columns={first_col: 'Capital'})

            cap_rep_cols = ['Unnamed: 1', 'Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4', 'Unnamed: 5']
            sml_rep_cols = ['Unnamed: 10', 'Unnamed: 11', 'Unnamed: 12', 'Unnamed: 13', 'Unnamed: 14']
            voters = ['Jash', 'Saimon', 'Ysa', 'Sean']

            print(f"\n{'='*55}")
            print(f"Worksheet {worksheet_num} - Voting Results")
            print(f"{'='*55}")

            any_printed      = False
            current_capital  = None
            current_small    = None
            letter_position  = 0
            letter_row_idx   = None
            voter_rows       = []

            def process_letter(letter, is_capital, num_incorrect_val, given_grade_val, v_rows, rep_cols):
                nonlocal any_printed

                if not pd.notna(num_incorrect_val):
                    return
                val_str = str(num_incorrect_val).strip()
                if val_str in ['', 'no grade']:
                    return
                try:
                    num_inc = int(float(val_str))
                except (ValueError, TypeError):
                    return

                if num_inc == 0:
                    return

                # Dynamically calculate majority from actual voter count
                num_voters = len(v_rows)
                majority   = (num_voters // 2) + 1  # 3 voters → 2, 4 voters → 3

                # A repetition is incorrect if majority of voters marked it as 1.0
                incorrect_indices = []
                for i, col in enumerate(rep_cols, start=1):
                    votes = [r[col] for r in v_rows if pd.notna(r[col])]
                    if sum(v == 1.0 for v in votes) >= majority:
                        incorrect_indices.append(i)

                if not incorrect_indices:
                    return

                any_printed = True
                case_label  = "Capital" if is_capital else "Small"
                print(f"  [{case_label}] Letter    : {letter}")
                print(f"  Position          : {letter_position}")
                print(f"  # Incorrect       : {num_inc}/5 repetitions")
                print(f"  Majority          : {majority}/{num_voters} voters needed")
                print(f"  Given Grade       : {given_grade_val}")
                print(f"  Incorrect Indices : {incorrect_indices}")
                print(f"  {'-'*45}")

            for idx, row in df.iterrows():
                capital = str(row['Capital']).strip()
                small   = str(row['Small Letters']).strip()

                if capital in list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'):
                    # Process previous letter block before moving to next
                    if current_capital and voter_rows:
                        prev_row = df.loc[letter_row_idx]
                        process_letter(current_capital, True,
                                       prev_row['No. of Incorrect'], prev_row['Given Grade'],
                                       voter_rows, cap_rep_cols)
                        process_letter(current_small, False,
                                       prev_row['# of Incorrect'], prev_row['Given Grade.1'],
                                       voter_rows, sml_rep_cols)

                    current_capital  = capital
                    current_small    = small if small in list('abcdefghijklmnopqrstuvwxyz') else capital.lower()
                    letter_position += 1
                    letter_row_idx   = idx
                    voter_rows       = []

                elif capital in voters:
                    voter_rows.append(row)

            # Process last letter block
            if current_capital and voter_rows:
                prev_row = df.loc[letter_row_idx]
                process_letter(current_capital, True,
                               prev_row['No. of Incorrect'], prev_row['Given Grade'],
                               voter_rows, cap_rep_cols)
                process_letter(current_small, False,
                               prev_row['# of Incorrect'], prev_row['Given Grade.1'],
                               voter_rows, sml_rep_cols)

            if not any_printed:
                print("  No letters marked incorrect by majority vote.")

            print()

        except Exception as e:
            import traceback
            print(f"Error reading worksheet {worksheet_num}: {e}")
            traceback.print_exc()

def main():
    # Initialize validator
    validator = WorksheetValidator('VotingPage.xlsx', threshold=2)
    
    # Example: Single worksheet validation
    worksheet_num = 1
    
    # Your AI-generated letter instances for this worksheet
    letter_instances = [
        {
            'letter': 'A',
            'repetition_number': 1,
            'position_in_worksheet': 1,
            'letter_form': 85.5,
            'size': 90.0,
            'line_align': 78.5,
            'bbox_x': 100,
            'bbox_y': 50,
            'bbox_width': 45,
            'bbox_height': 60,
            'comments': 'Good formation'
        },
        {
            'letter': 'B',
            'repetition_number': 1,
            'position_in_worksheet': 2,
            'letter_form': 65.0,  # Low score - might be incorrect
            'size': 70.0,
            'line_align': 68.0,
            'bbox_x': 150,
            'bbox_y': 50,
            'bbox_width': 45,
            'bbox_height': 60,
            'comments': 'Needs improvement'
        },
        # ... more letters
    ]
    
    # Calculate confusion matrix for one worksheet
    # metrics = validator.calculate_confusion_matrix(worksheet_num, letter_instances)
    #
    # print(f"\nWorksheet {worksheet_num} Validation:")
    # for metric in metrics:
    #     print(f"Letter {metric.letter}: TP={metric.true_positives}, "
    #           f"TN={metric.true_negatives}, FP={metric.false_positives}, "
    #           f"FN={metric.false_negatives}, Accuracy={metric.accuracy():.2%}")
    #
    # # Generate report for all worksheets
    # all_worksheet_data = {
    #     1: letter_instances,  # Worksheet 1 data
    #     2: letter_instances,  # Worksheet 2 data
    # }
    #
    # report_df = validator.generate_validation_report(all_worksheet_data)
    # print("\n" + "="*80)
    # print("VALIDATION REPORT")
    # print("="*80)
    # print(report_df)
    #
    # # Summary statistics
    # summary = validator.generate_summary_statistics(report_df)
    # print("\n" + "="*80)
    # print("SUMMARY STATISTICS")
    # print("="*80)
    # for key, value in summary.items():
    #     if isinstance(value, float):
    #         print(f"{key}: {value:.2%}")
    #     else:
    #         print(f"{key}: {value}")
    #
    # # Save report
    # report_df.to_excel('validation_report.xlsx', index=False)
    # print("\nReport saved to validation_report.xlsx")
    #
    validator = WorksheetValidator('VotingPage.xlsx', threshold=2)

# Single worksheet
    validator.print_voting_results(2)

if __name__ == "__main__":
    main()
