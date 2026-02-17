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
            # First image - just assign directly
            self.letter_summary = new_rep
            return

        # letter_instances: extend the list
        if 'letter_instances' in new_rep:
            self.letter_summary['letter_instances'].extend(new_rep['letter_instances'])

        # letter_summaries: extend the list
        if 'letter_summaries' in new_rep:
            self.letter_summary['letter_summaries'].extend(new_rep['letter_summaries'])

        # worksheet_summary: overwrite with latest (or you could average them)
        # if 'worksheet_summary' in new_rep:
        #     self.letter_summary['worksheet_summary'] = new_rep['worksheet_summary']

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
