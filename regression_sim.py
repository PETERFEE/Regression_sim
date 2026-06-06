import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from matplotlib.widgets import Button, RadioButtons

# 1. Initialize data storage and state
x_data = []
y_data = []
current_mode = 'Regression'
current_degree = 1

# 2. Set up the main plot and widget areas
fig, ax = plt.subplots(figsize=(10, 7))
plt.subplots_adjust(bottom=0.35) # Make room for widgets at the bottom
ax.set_title('Overfitting Simulation: Click to add points')
ax.set_xlim(0, 30)
ax.set_ylim(-10, 1010)
ax.grid(True, linestyle='--', alpha=0.6)

# Initialize plot objects
scatter_plot, = ax.plot([], [], 'o', color='#0099ff', markersize=8, label='Data')
line_plot, = ax.plot([], [], 'b-', linewidth=3, label='Model Fit')
ax.legend(loc='lower right')

# 3. Define Widget Axes [left, bottom, width, height]
ax_radio_mode = plt.axes([0.15, 0.05, 0.15, 0.15])
ax_radio_deg = plt.axes([0.35, 0.05, 0.15, 0.20])
ax_btn_fit = plt.axes([0.55, 0.15, 0.12, 0.08])
ax_btn_clear = plt.axes([0.55, 0.05, 0.12, 0.08])

# 4. Create Widgets
radio_mode = RadioButtons(ax_radio_mode, ('Regression', 'Categorical'))
radio_deg = RadioButtons(ax_radio_deg, ('1', '2', '3', '4', '5', '6'))
btn_fit = Button(ax_btn_fit, 'Fit Data', color='#0099ff', hovercolor='#0077cc')
btn_clear = Button(ax_btn_clear, 'Clear Data', color='#ff4c4c', hovercolor='#cc0000')

# 5. Event Handlers
def mode_changed(label):
    global current_mode
    current_mode = label
    # Categorical mode usually works with 0 and 1, so adjust axes purely for visuals
    if current_mode == 'Categorical':
        ax.set_ylim(-0.1, 1.1)
    else:
        ax.set_ylim(-10, 1010)
    fig.canvas.draw_idle()

def deg_changed(label):
    global current_degree
    current_degree = int(label)

def onclick(event):
    if event.inaxes != ax: return
    
    x_val = event.xdata
    y_val = event.ydata
    
    # If categorical, snap the Y value to 0 or 1 based on where the user clicked
    if current_mode == 'Categorical':
        y_val = 1 if y_val > 0.5 else 0

    x_data.append(x_val)
    y_data.append(y_val)
    scatter_plot.set_data(x_data, y_data)
    fig.canvas.draw_idle()

def clear_data(event):
    x_data.clear()
    y_data.clear()
    scatter_plot.set_data([], [])
    line_plot.set_data([], [])
    ax.set_title('Overfitting Simulation: Click to add points')
    fig.canvas.draw_idle()

def fit_model(event):
    if len(x_data) < 2: return
    
    X = np.array(x_data).reshape(-1, 1)
    y = np.array(y_data)
    
    # Apply Polynomial Features
    poly = PolynomialFeatures(degree=current_degree)
    X_poly = poly.fit_transform(X)
    
    x_smooth = np.linspace(0, 30, 200).reshape(-1, 1)
    x_smooth_poly = poly.transform(x_smooth)

    try:
        if current_mode == 'Regression':
            model = LinearRegression()
            model.fit(X_poly, y)
            y_pred = model.predict(x_smooth_poly)
            ax.set_title(f'Regression Fit (Degree {current_degree})')
            
        elif current_mode == 'Categorical':
            # Logistic Regression expects binary classes
            # If all points are the same class, it will fail, so we check first
            if len(set(y)) < 2:
                ax.set_title('Need both Class 0 and Class 1 for Logistic Fit!')
                return
                
            # Increase max_iter for higher degree polynomials to converge
            model = LogisticRegression(max_iter=10000, solver='lbfgs')
            model.fit(X_poly, y)
            y_pred = model.predict_proba(x_smooth_poly)[:, 1] # Plot the probability curve
            ax.set_title(f'Logistic Probability Curve (Degree {current_degree})')

        line_plot.set_data(x_smooth, y_pred)
        fig.canvas.draw_idle()
    except Exception as e:
        print(f"Fit error: {e}")

# 6. Connect Events
radio_mode.on_clicked(mode_changed)
radio_deg.on_clicked(deg_changed)
btn_fit.on_clicked(fit_model)
btn_clear.on_clicked(clear_data)
fig.canvas.mpl_connect('button_press_event', onclick)

plt.show()