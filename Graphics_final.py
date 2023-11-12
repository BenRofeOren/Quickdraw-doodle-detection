import os

import pygame
from PIL import Image  # remember to install the library Pillow

import tensorflow
from tensorflow.keras.models import load_model


class Window:
    def __init__(self):
        pygame.init()
        self.presets_dir = "Presets"  # the directory in which we store the saved images
        if not os.path.exists(self.presets_dir):   # if the directory "Presets" does not exist, create it
            os.mkdir(self.presets_dir)
        self.presets = []  # list for the preset images
        for directory in os.listdir(self.presets_dir):
            self.presets.append((directory[:-4], Image.open(self.presets_dir + "\\" + directory)))
        self.offset = 0  # offset from the start of the list on the left_bar of the screen

        # colors for the three areas of the window
        self.background_color = (255, 255, 255)
        self.right_bar_color = (255, 128, 0)
        self.left_bar_color = (101, 255, 192)
        window_size = (500, 300)  # choosing the window size
        self.screen = pygame.display.set_mode(window_size)
        pygame.display.set_caption('Image recognition')
        self.screen.fill(self.background_color)
        # 3 rectangles representing the 3 areas of the window, the left bar, the drawing_board and the right bar
        self.drawing_board = pygame.draw.rect(self.screen, self.background_color, pygame.Rect(100, 0, 300, 300))
        self.right_bar = pygame.draw.rect(self.screen, self.right_bar_color, pygame.Rect((400, 0, 100, 300)))
        self.left_bar = pygame.draw.rect(self.screen, self.left_bar_color, pygame.Rect((0, 0, 100, 300)))

        pygame.display.flip()  # this is intended to update the screen

        self.font = pygame.font.SysFont('freesansbold.ttf', 15)
        self.left_bar_font = pygame.font.SysFont('freesansbold.ttf', 20)

    # clears the drawing area of the window
    def clear_screen(self):
        pygame.draw.rect(self.screen, self.background_color, self.drawing_board)

    # refreshes the right bar of the window with percentage list, if percentage list is
    # not given it only refreshes the rectangle with no list
    def refresh_rightbar(self, percentage_list: list = None):
        # redraws the sidebar
        pygame.draw.rect(self.screen, self.right_bar_color, self.right_bar)
        # if you give it a list to project
        if percentage_list is not None:
            # sorting from Google
            sorted_list = sorted(percentage_list, key=lambda x: x[1], reverse=True)
            # get first 5 items (first = strong,big,great last = weak,small,bad)
            final_list = sorted_list[:5]
            for i in range(5 * 2):
                if i % 2 == 0:
                    text_temp = str(final_list[round(i / 2)][0])
                    if final_list[round(i / 2)][0] == "The Eiffel Tower":
                        text_temp = "Eiffel Tower"
                else:
                    text_temp = str(round(final_list[round(i / 2)][1], 5))
                # make text from item and percentage
                text = self.font.render(text_temp, True, (0, 0, 0))
                # text = self.font.render(final_list[i][0] + ' ' + str(final_list[i][1]), True, (0, 0, 0))
                # get its rect
                text_rect = text.get_rect()
                # place center where I want it to be (7/8ths of screen width, 25 + 50 space per text)
                text_rect.center = (self.right_bar.center[0], 25 + 50 * i / 2)
                # draw text on to sidebar
                self.screen.blit(text, text_rect)

    # refreshes the left bar of the window, if the direction is not 0 it will scroll the preset
    # list by adding or subtracting offset
    def refresh_leftbar(self, direction=0):
        if direction != 0 and 0 <= self.offset + direction <= len(self.presets) - 5:
            self.offset += direction
        pygame.draw.rect(self.screen, self.left_bar_color, self.left_bar)
        # for i in range(self.offset, (5 + self.offset if len(self.presets) >= 5 else len(self.presets))):
        for i in range((5 if len(self.presets) >= 5 else len(self.presets))):
            text_temp = str(self.presets[i+self.offset][0])
            # make text from item and percentage
            text = self.left_bar_font.render(text_temp, True, (0, 0, 0))
            # text = self.font.render(final_list[i][0] + ' ' + str(final_list[i][1]), True, (0, 0, 0))
            # get its rect
            text_rect = text.get_rect()
            # place center where I want it to be (7/8ths of screen width, 25 + 50 space per text)
            text_rect.center = (self.left_bar.center[0], 50 + 50 * i)
            # draw text on to sidebar
            self.screen.blit(text, text_rect)

    # returns PIL.Image in size compress_size in order to use the neural network
    # if cut_edges is true it will trim away any extra pure white edges
    def get_screen(self, compress_size=(64, 64), cut_edges=False):
        # get the drawing area of the screen
        sub_surface = self.screen.subsurface(pygame.Rect(100, 0, 300, 300))
        # get the pixel data as an array from the subsurface
        pixel_data = pygame.surfarray.array3d(sub_surface)
        # convert the pixel data into a PIL Picture
        img = Image.fromarray(pixel_data)

        if cut_edges:
            nonwhite_positions = [(x, y) for x in range(img.size[0]) for y in range(img.size[1]) if img.getdata()[x + y * img.size[0]] != (255, 255, 255)]
            if len(nonwhite_positions) != 0:
                rect = (min([x for x, y in nonwhite_positions]), min([y for x, y in nonwhite_positions]), max([x for x, y in nonwhite_positions]), max([y for x, y in nonwhite_positions]))
                img = img.crop(rect)

        output_data = img.resize(compress_size).transpose(Image.TRANSPOSE)
        return output_data

    # predict the image currently drawn and update the right bar
    def make_prediction(self):
        tensor = tensorflow.convert_to_tensor(self.get_screen(cut_edges=True))
        tensor = tensorflow.image.resize(tensor, [64, 64])
        input_tensor = tensorflow.expand_dims(tensor, axis=0)
        results = ai.__call__(input_tensor, training=False)
        results_list = tensorflow.Variable(results).numpy().tolist()
        results_list = results_list[0]
        categories = ['airplane', 'apple', 'axe', 'basketball', 'bicycle', 'broccoli', 'bucket', 'butterfly',
                      'crab', "diamond",
                      'fence', 'fish', "guitar", 'hammer', 'headphones', 'helicopter', 'hot air balloon',
                      'ice cream',
                      'light bulb', 'lollipop', 'palm tree', 'parachute', 'rainbow', 'sailboat', 'shoe',
                      'smiley face', 'star',
                      "tennis racquet", 'traffic light', 'wristwatch']
        prediction = dict(zip(results_list, categories))
        # sorted_prediction = sorted(prediction)
        prediction_to_print = []
        for key in prediction.keys():
            prediction_to_print.append((prediction[key], key))
        self.refresh_rightbar(prediction_to_print)

    # window loop
    def run(self):
        running = True
        last_mouse_x, last_mouse_y = None, None
        last_left, last_middle, last_right = None, None, None
        last_keys = None
        self.refresh_leftbar()
        drawing_hold = None
        history = []
        while running:
            wheel_y = 0
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.MOUSEWHEEL:
                    wheel_y = event.y

            mouse_x, mouse_y = pygame.mouse.get_pos()
            left, middle, right = pygame.mouse.get_pressed()
            keys = pygame.key.get_pressed()

            # if mouse was pressed but not held it saves the image
            if left and not last_left:
                drawing_hold = self.get_screen((300, 300))

            # if mouse was released it appends to history the saved image and if history is too big trims it.
            if last_left and not left:
                history.append(drawing_hold)
                if len(history) > 15:
                    history.pop(0)
                drawing_hold = None

            # is z is pressed not held and there is something to restore it will restore the last saved image
            if keys[pygame.K_z] and not last_keys[pygame.K_z] and len(history) > 0:
                to_load = history[-1]
                history.pop(-1)
                pygame_surface = pygame.image.fromstring(to_load.tobytes(), to_load.size, "RGB").convert()
                self.screen.blit(pygame_surface, pygame_surface.get_rect(center=self.drawing_board.center))
                self.make_prediction()

            # is s is pressed but not held it will save the image currently drawn to presets with a name given in console
            if keys[pygame.K_s] and not last_keys[pygame.K_s]:
                file_name = input("name the picture:\n")
                self.get_screen((300, 300)).save('Presets\\' + file_name + '.png')
                self.presets = []
                for directory in os.listdir(self.presets_dir):
                    self.presets.append((directory[:-4], Image.open(self.presets_dir + "\\" + directory)))
                self.refresh_leftbar()

            # if the mouse is over the left bar of the window and is scrolling refresh the left bar
            if mouse_x < 100 and wheel_y != 0:
                self.refresh_leftbar(-wheel_y)

            # if left mouse was clicked but not held and is over the left_bar load the image being clicked
            if left and not last_left and mouse_x < 100:
                row = (mouse_y - 25) // 50
                if row < 5:
                    to_load = self.presets[row + self.offset][1].resize((300, 300)).convert("RGB")
                    pygame_surface = pygame.image.fromstring(to_load.tobytes(), to_load.size, "RGB").convert()
                    self.screen.blit(pygame_surface, pygame_surface.get_rect(center=self.drawing_board.center))
                    self.make_prediction()

            # if the mouse is on the drawing area and left click is clicked or held draw a line from the last location and the current one
            if last_mouse_x is not None and 100 <= last_mouse_x < 400 and last_mouse_y is not None and left:
                pygame.draw.line(self.screen, (0, 0, 0), (mouse_x, mouse_y), (last_mouse_x, last_mouse_y), 6)
                self.refresh_leftbar()

            # if mouse is over the drawing area of the window and released predict the image that was drawn
            if last_mouse_x is not None and 100 <= last_mouse_x < 400 and last_mouse_y is not None and not left and last_left:
                self.make_prediction()

            # if right click was clicked not held
            if right and not last_right:
                history.append(self.get_screen((300, 300)))
                if len(history) > 15:
                    history.pop(0)
                self.clear_screen()

            last_mouse_x, last_mouse_y = mouse_x, mouse_y
            last_left, last_middle, last_right = left, middle, right
            last_keys = keys
            pygame.display.update()


if __name__ == '__main__':
    # load model
    # ai = load_model('transfer_dropout_batch-norm.h5')
    ai = load_model("models/final_model.h5", compile=False)
    print(['airplane', 'apple', 'axe', 'basketball', 'bicycle', 'broccoli', 'bucket', 'butterfly',
                      'crab', "diamond",
                      'fence', 'fish', "guitar", 'hammer'])
    print(['headphones', 'helicopter', 'hot air balloon',
                      'ice cream',
                      'light bulb', 'lollipop', 'palm tree', 'parachute', 'rainbow', 'sailboat', 'shoe'])
    print(['smiley face', 'star',
                      "tennis racquet", 'traffic light', 'wristwatch'])
    window = Window()
    window.run()
