class Report:

    def __init__(self, folder_name: str ="") -> None:
        '''
        Args:
            folder_name: name of the folder
              grade_lvl: grade_lvl of the worksheet
        '''
        self.folder_name = folder_name
        self.grade_lvl = self.set_grade_lvl()
        self.ws_num = self.set_ws_num()
        self.letter_summary = {
            'letter_instances':  [],
            'letter_summaries':  [],
            'worksheet_summary': {}
        }

    def set_grade_lvl(self):
        return self.folder_name.split(' - ')[1].strip().replace(' ', '').lower()

    def set_ws_num(self):
        return self.folder_name.split(' - ')[0].strip().replace(' ', '').lower()

    def append_report(self, new_rep: dict):
        if not self.letter_summary:
            self.letter_summary = new_rep
            return

        if 'letter_instances' in new_rep:
            existing = self.letter_summary['letter_instances']
            existing_keys = {
                (inst['letter'], inst['repetition_num'])
                for inst in existing
            }
            for inst in new_rep['letter_instances']:
                key = (inst['letter'], inst['repetition_num'])
                if key not in existing_keys:
                    existing.append(inst)
                    existing_keys.add(key)

        if 'letter_summaries' in new_rep:
            existing_summaries = {
                s['letter']: s
                for s in self.letter_summary['letter_summaries']
            }

            # Group incoming summaries by their character range
            new_summaries_by_letter = {s['letter']: s for s in new_rep['letter_summaries']}
            new_letters = set(new_summaries_by_letter.keys())

            # Find which existing letters overlap with incoming
            overlapping_letters = new_letters & set(existing_summaries.keys())

            if overlapping_letters:
                # Check if ALL overlapping letters have a higher score in new_rep
                all_higher = all(
                    new_summaries_by_letter[letter]['letter_average'] >
                    existing_summaries[letter]['letter_average']
                    for letter in overlapping_letters
                )

                if all_higher:
                    old_avg = sum(existing_summaries[l]['letter_average'] for l in overlapping_letters) / len(overlapping_letters)
                    new_avg = sum(new_summaries_by_letter[l]['letter_average'] for l in overlapping_letters) / len(overlapping_letters)
                    print(f"↑ All {sorted(overlapping_letters)} scores higher — updating: {old_avg:.2f} → {new_avg:.2f}")
                    for letter in overlapping_letters:
                        existing_summaries[letter] = new_summaries_by_letter[letter]
                else:
                    # Find which ones were not higher for visibility
                    not_higher = [
                        l for l in overlapping_letters
                        if new_summaries_by_letter[l]['letter_average'] <=
                           existing_summaries[l]['letter_average']
                    ]
                    print(f"↓ Keeping existing — not all scores higher, failed on: {sorted(not_higher)}")
            
            # Add any brand new letters not previously seen
            for letter in new_letters - overlapping_letters:
                print(f"+ Adding new letter '{letter}'")
                existing_summaries[letter] = new_summaries_by_letter[letter]

            # Write back as list
            self.letter_summary['letter_summaries'] = list(existing_summaries.values())

        summaries = self.letter_summary['letter_summaries']
        instances = self.letter_summary['letter_instances']
        avg_form  = sum(s['avg_letter_form'] for s in summaries) / len(summaries)
        avg_size  = sum(s['avg_size']        for s in summaries) / len(summaries)
        avg_align = sum(s['avg_line_align']  for s in summaries) / len(summaries)
        overall   = (avg_form + avg_size + avg_align) / 3
        self.letter_summary['worksheet_summary'] = {
            **self.letter_summary.get('worksheet_summary', {}),
            'overall_letter_form': round(avg_form, 2),
            'overall_size':        round(avg_size, 2),
            'overall_line_align':  round(avg_align, 2),
            'overall_score':       round(overall, 2),
            'total_letters':       len(summaries),
            'total_repetitions':   len(instances),
        }

    def generate_report(self):
        print("Generating report")

class ReportGenerator:

    def __init__(self, report: Report, sheet: str) -> None:
        '''
        Args:
            report: an instance of class Report containing reports for each image in a folder
             sheet: directory of the sheet for cross referencing and generate TP, FP, TN, FN
        '''
        self.report = report
        self.sheet  = sheet 
