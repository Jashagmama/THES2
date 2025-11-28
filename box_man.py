'''
This class initializes all the boxes on a worksheet based on the coordinates
'''

class Box:
    x = 0
    y = 0
    w = 0
    h = 0

    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

class Letter:
    def __init__(self, char, box: Box, is_template=False):
        self.letter_form = 0     # character confidence level of letter
        self.orientation = 0.0    # skew of letter
        self.size = 0           # height of letter?
        self.line_align = 0     # y coordinate of the lower part of letter
        self.char = char
        self.box = box
        self.letter_form_status = False
        self.line_align_status = False
        self.size_status = False
        self.orientation_status = False
        self.is_template = is_template


    def isPass(self):
        return (self.letter_form_status and self.line_align_status and
                self.size_status and self.orientation_status)

    def print_coords(self):
        print(f"x: {self.box.x}, y: {self.box.y}, w: {self.box.w}, h: {self.box.h}")

    def print_status(self):
        print(f'character: {self.char}')
        print(f'letter_form_status: {self.letter_form_status}')
        print(f'line_align_status: {self.line_align_status}')
        print(f'size_status: {self.size_status}')
        print(f'orientation_status: {self.orientation_status}')

    def print_vals(self):
        print(f'letter_form: {self.letter_form}')
        print(f'orientation: {self.orientation}')
        print(f'size: {self.size}')
        print(f'line_align: {self.line_align}')

class Boxman():
    # x,y,w,h format

    def __init__(self, mode="check"):
        self.cells = []
        self.letters = []
        print(f"mode: {mode}")
        mode = mode.strip()
        # if (mode == "main"):
        #     coords = [
        #         [4,5,124,99], [127,5,127,100], [252,5,129,100], [381,5,129,100], [510,5,129,100], [640,5,129,100],
        #         # [5,104,122,101], [127,104,124,101], [254,104,125,101], [382,104,126,101], [512,104,126,101], [641,104,126,101],
        #         # [5,204,119,100], [130,204,121,97], [255,204,124,97], [385,204,124,97], [513,204,125,97], [643,204,124,97],
        #         # [5,301,119,98], [129,302,121,97], [255,304,123,95], [384,304,123,95], [513,304,123,95], [643,304,124,96],
        #         # [6,400,117,98], [129,400,121,98], [255,404,124,94], [384,404,123,94], [514,404,123,94], [644,404,123,94],
        #         # [6,499,117,99], [129,499,121,99], [255,499,124,99], [384,499,124,99], [514,501,124,95], [644,501,123,95],
        #         # [6,601,117,95], [129,600,121,97], [255,600,123,98], [384,600,123,98], [514,600,123,98], [644,600,123,98],
        #         # [6,703,117,95], [129,702,120,95], [255,702,123,95], [384,702,124,95], [514,702,124,95], [644,702,124,95],
        #         # [6,800,117,95], [129,800,119,94], [254,802,124,90], [384,802,124,90], [514,802,124,90], [644,802,124,90],
        #         # [6,899,116,96], [128,898,120,96], [254,898,124,96], [385,898,123,96], [515,898,123,96], [644,898,123,96],
        #     ]
        #
        # elif (mode == "check"):
        #     coords = [
        #         [51,121,219,161], [51,291,219,171],
        #         # [53,470,217,166]
        #     ]
        # else:
        #     coords = []
        match mode:
            case "check":
                # Coordinates is based on non-cropped only after SIFT
                # Coordinates only existing of the first two rows at column 1 item so, the first two template chars
                coords = [
                    [51,121,219,161], [51,291,219,171] # First two rows, [53,470,217,166]
                    # [53,471,217,164], [53,643,217,167] # 3rd and 4th row
                ]
            case "main":
                # Coordinates is based on cropped to grid locating
                coords = [
                    [4,5,124,99], [127,5,127,100], [252,5,129,100], [381,5,129,100], [510,5,129,100], [640,5,129,100],
                    [5,104,122,99], [127,104,124,101], [254,104,125,101], [382,104,126,101], [512,104,126,101], [641,104,126,101],
                    [5,204,119,98], [130,204,121,97], [255,204,124,97], [385,204,124,97], [513,204,125,97], [643,204,124,97],
                    [5,301,119,98], [129,302,121,97], [255,304,123,95], [384,304,123,95], [513,304,123,95], [643,304,124,96],
                    [6,400,117,98], [129,400,121,98], [255,404,124,94], [384,404,123,94], [514,404,123,94], [644,404,123,94],
                    [6,499,117,99], [129,499,121,99], [255,499,124,99], [384,499,124,99], [514,501,124,95], [644,501,123,95],
                    [6,601,117,95], [129,600,121,97], [255,600,123,98], [384,600,123,98], [514,600,123,98], [644,600,123,98],
                    [6,703,117,95], [129,702,120,95], [255,702,123,95], [384,702,124,95], [514,702,124,95], [644,702,124,95],
                    [6,800,117,95], [129,800,119,94], [254,802,124,90], [384,802,124,90], [514,802,124,90], [644,802,124,90],
                    [6,899,116,96], [128,898,120,96], [254,898,124,96], [385,898,123,96], [515,898,123,96], [644,898,123,96],
                ]
            case "partial":
                coords = [ # y,z only template
                    [4,5,124,99], [127,5,127,100], [252,5,129,100], [381,5,129,100], [510,5,129,100], [640,5,129,100],
                    [5,104,122,101], [127,104,124,101], [254,104,125,101], [382,104,126,101], [512,104,126,101], [641,104,126,101],
                ]
            case _:
                print("Cannot initialize coords")
                coords = []

        for x,y,w,h in coords:
            self.cells.append( Box(x,y,w,h) )

    def print_all(self):
        for cell in self.cells:
            print(cell.x)

