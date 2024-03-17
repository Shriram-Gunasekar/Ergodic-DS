from PyDSTool import args, Generator, points, integrate, intersect
import matplotlib.pyplot as plt

# Define the logistic map system
def logistic_map(ic=None, pars=None, tdata=None):
    x = args(name='x')
    eqns = [x - pars['r'] * x * (1 - x)]
    DSargs = args(name='LogisticMap')
    DSargs.pars = pars
    DSargs.tdata = tdata
    DSargs.varspecs = {'x': eqns}
    DSargs.ics = {'x': ic}
    return Generator.Vode_ODEsystem(DSargs)

# Define the bifurcation analysis function
def bifurcation_analysis(logistic_map, r_values, ic, tdata):
    bifurcation_data = []
    for r in r_values:
        logistic_map.set(pars={'r': r})
        traj = logistic_map.compute('demo')
        pts = traj.sample()
        if pts[0]['x'] != pts[-1]['x']:  # Check for a closed orbit (limit cycle)
            bp = pts[-1]  # Last point of the orbit
            bifurcation_data.append((r, bp['x']))
    return zip(*bifurcation_data)

# Parameters
r_values = np.linspace(2.5, 4.0, 1000)  # Range of parameter values (r)
ic = {'x': 0.1}  # Initial condition
tdata = [0, 100]  # Time span for simulation

# Create the logistic map system
logistic_map = logistic_map(ic=ic, pars={'r': 0}, tdata=tdata)

# Perform bifurcation analysis
r_values, x_values = bifurcation_analysis(logistic_map, r_values, ic, tdata)

# Plot the bifurcation diagram
plt.figure(figsize=(10, 6))
plt.plot(r_values, x_values, '.', markersize=1, label='Bifurcation Diagram')
plt.xlabel('r')
plt.ylabel('x')
plt.title('Bifurcation Diagram for Logistic Map')
plt.grid(True)
plt.legend()
plt.show()
