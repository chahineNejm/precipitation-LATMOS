import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox, Button
from matplotlib import patches
from DBSCAN_utils import data_importation  # Ensure this is correctly imported

# Data importation
RR = data_importation(year=2018, day=6, month=2)  # Replace with your actual function

# Setup figure and axes for the main plot, the slider, and the widgets
fig, ax = plt.subplots(1, 2, figsize=(15, 6))
plt.subplots_adjust(bottom=0.45)  # Make more space for widgets

# Display an example image on the first subplot
cmap = ax[0].imshow(np.sum(RR, axis=0), cmap='viridis', interpolation='nearest')
fig.colorbar(cmap, ax=ax[0], orientation='vertical')
ax[0].set_title("Click on a point in the grid")

# Adding a patch object which will be used to show the clicked location
click_indicator = patches.Circle((0, 0), radius=5, color='red', fill=False)
ax[0].add_patch(click_indicator)

# Line plot for the time series on the second subplot
line, = ax[1].plot(RR[:, 150, 150])
ax[1].set_title("Time Series at (150, 150)")
ax[1].set_xlabel("Time Index")
ax[1].set_ylabel("rain (mm)")
ax[1].grid(True)

# Create axes for slider
ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03])
slider = Slider(ax=ax_slider, label='Time Slider', valmin=0, valmax=100, valinit=50)

# Create axes for first and second text boxes
ax_textbox1 = plt.axes([0.25, 0.25, 0.3, 0.05])
text_box1 = TextBox(ax=ax_textbox1, label='X Position:', initial="150")
ax_textbox2 = plt.axes([0.6, 0.25, 0.3, 0.05])
text_box2 = TextBox(ax=ax_textbox2, label='Y Position:', initial="150")

# Create axes for buttons
ax_button_x_plus = plt.axes([0.25, 0.2, 0.15, 0.04])
btn_x_plus = Button(ax_button_x_plus, 'east')
ax_button_x_minus = plt.axes([0.4, 0.2, 0.15, 0.04])
btn_x_minus = Button(ax_button_x_minus, 'west')
ax_button_y_plus = plt.axes([0.6, 0.2, 0.15, 0.04])
btn_y_plus = Button(ax_button_y_plus, 'south')
ax_button_y_minus = plt.axes([0.75, 0.2, 0.15, 0.04])
btn_y_minus = Button(ax_button_y_minus, 'north')

# Global variables to store the x and y positions
x_pos = 150
y_pos = 150

def update_position():
    click_indicator.center = (x_pos, y_pos)
    new_series = RR[:, y_pos, x_pos]
    line.set_ydata(new_series)
    ax[1].set_title(f"Time Series at ({x_pos}, {y_pos})")
    fig.canvas.draw_idle()

def increment_x(event):
    global x_pos
    x_pos += 1
    update_position()

def decrement_x(event):
    global x_pos
    x_pos -= 1
    update_position()

def increment_y(event):
    global y_pos
    y_pos += 1
    update_position()

def decrement_y(event):
    global y_pos
    y_pos -= 1
    update_position()

def submit_x(text):
    global x_pos
    try:
        x_pos = int(text)
        update_position()
    except ValueError:
        print("Please enter a valid integer for X Position.")

def submit_y(text):
    global y_pos
    try:
        y_pos = int(text)
        update_position()
    except ValueError:
        print("Please enter a valid integer for Y Position.")
        
def onclick(event):
    
    global x_pos, y_pos
    if event.inaxes == ax[0]:
        ix, iy = int(round(event.xdata)), int(round(event.ydata))
        if 0 <= ix < RR.shape[2] and 0 <= iy < RR.shape[1]:
            x_pos, y_pos = ix, iy
            new_series = RR[:, iy, ix]
            line.set_ydata(new_series)
            ax[1].set_title(f"Time Series at ({ix}, {iy})")
            click_indicator.center = (ix, iy)
            fig.canvas.draw_idle()
        else:
            print("Clicked outside of valid data range.")

fig.canvas.mpl_connect('button_press_event', onclick)
btn_x_plus.on_clicked(increment_x)
btn_x_minus.on_clicked(decrement_x)
btn_y_plus.on_clicked(increment_y)
btn_y_minus.on_clicked(decrement_y)
text_box1.on_submit(submit_x)
text_box2.on_submit(submit_y)


plt.show()




