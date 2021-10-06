from matplotlib import pyplot as plt
from math import isinf
import numpy as np
from matplotlib.gridspec import GridSpec
import jinja2
import base64
from io import BytesIO
import pathlib
import sys
import webbrowser

img_width = 13

def show(q, qd, qdd, u, T, ee_pos, q_limits, steps, cost_func, final_term, constr, f_constr, show=False):
    # Defining the X axis for most cases
    tgrid = [T / steps * k for k in range(steps + 1)]
    tgrid = np.squeeze(np.array(tgrid))
    # Plotting Q and its derivatives
    fig1 = plot_q(q, qd, qdd, q_limits, u, tgrid)
    fig2 = plot_cost(q, qd, qdd, ee_pos, u, cost_func,final_term, tgrid)
    fig3 = plot_constraints(q, qd, qdd, ee_pos, u, constr, tgrid)
    final_results = eval_final_constr(q, qd, qdd, ee_pos, u, f_constr)
    if show:
        generate_html(fig1, fig2, fig3, final_results)
    return [fig1, fig2, *fig3], f_constr

def plot_q(q, qd, qdd, q_limits, u, tgrid):
    n_joints = len(q)
    # Setting for the plot axis
    gridspec_kw={   'width_ratios': [2, 1, 1, 1],
                    'wspace': 0.4,
                    'hspace': 0.4}

    fig, axes = plt.subplots(nrows=n_joints, ncols=4, figsize=(img_width,2.2*n_joints), gridspec_kw=gridspec_kw)
    fig.suptitle('Joints and Inputs', fontsize=14)

    if n_joints==1: axes = [axes] # make it a list for enumerate
    for idx, ax in enumerate(axes):
        # Painting the boundaries
        lb, ub = q_limits['q'][0][idx], q_limits['q'][1][idx]
        if not isinf(lb) and not isinf(ub):
            ax[0].set_facecolor((1.0, 0.45, 0.4))
            ax[0].axhspan(lb, ub, facecolor='w')

        # Plotting the values
        ax[0].plot(tgrid, q[idx], '-')
        ax[0].legend('q'+str(idx))
        ax[0].set_xlabel('time')
        ax[0].set_ylabel('q'+str(idx))
        if idx == 0: ax[0].set_title('q plot')
        ax[0].grid()

        # Painting the boundaries
        lb, ub = q_limits['qd'][0][idx], q_limits['qd'][1][idx]
        if not isinf(lb) and not isinf(ub):
            ax[1].set_facecolor((1.0, 0.45, 0.4))
            ax[1].axhspan(lb, ub, facecolor='w')

        ax[1].plot(tgrid, qd[idx], 'g-')
        ax[1].legend('qd'+str(idx))
        ax[1].set_xlabel('time')
        if idx == 0: ax[1].set_title('qd plot')
        ax[1].grid()

        ax[2].plot(tgrid[:-1], qdd[idx], 'y-')
        ax[2].legend('qdd'+str(idx))
        ax[2].set_xlabel('time')
        if idx == 0: ax[2].set_title('qdd plot')
        ax[2].grid()

        lb, ub = q_limits['u'][0][idx], q_limits['u'][1][idx]
        if not isinf(lb) and not isinf(ub):
            ax[3].set_facecolor((1.0, 0.45, 0.4))
            ax[3].axhspan(lb, ub, facecolor='w')

        ax[3].plot(tgrid[:-1], u[idx], 'g-')
        ax[3].legend('ud_'+str(idx))
        ax[3].set_xlabel('time')
        if idx == 0: ax[3].set_title('u plot')
        ax[3].grid()
    return fig

def plot_cost(q, qd, qdd, ee_pos, u, cost_func,final_term, tgrid):
    n_joints = len(q)
    # Setting for the axis
    gridspec_kw={   'wspace': 0.4,
                    'hspace': 0.4}
    
    # Instantiating plot
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(img_width,4), gridspec_kw=gridspec_kw)
    fig.suptitle('Cost Function', fontsize=14)

    # Refactor function for q and its derivatives to be [q0(0), q1(0), q2(0)],[q0(1), q1(1), q2(1)] etc
    ref = lambda q : [np.array([q[j][idx] for j in range(n_joints)]) for idx in range(len(q[0]))]

    # Calculating the value of the cost function for each point
    iterator = zip(ref(q), ref(qd), ref(qdd), ref(ee_pos), ref(u), tgrid)
    cost_plot = np.array([cost_func(q,qd,qdd,ee_pos,u,t) for q,qd,qdd,ee_pos,u,t in iterator])
    cost_plot = np.squeeze(cost_plot)

    # Final Value
    cumulated_cost = [sum(cost_plot[:idx+1]) for idx in range(len(cost_plot))]
    
    if final_term is not None:
        # Storing the variable for the final term cost
        qf = [q[i][-1] for i in range(n_joints)]
        qdf = [qd[i][-1] for i in range(n_joints)]
        qddf = [qdd[i][-1] for i in range(n_joints)]
        ee_posf = [ee_pos[i][-1] for i in range(3)]
        uf = [u[i][-1] for i in range(n_joints)]
        # Evaluating the final term cost
        final_cost = float(final_term(qf,qdf,qddf,ee_posf,uf))
    else:
        final_cost = 0

    # Plotting both final term cost and the cost function
    legend = ['cumulated cost function', 'cost function']
    axes.plot(tgrid[:-1], cumulated_cost, '-', color='tab:orange')
    axes.fill_between(tgrid[:-1], cumulated_cost, color='tab:orange') 
    axes.plot(tgrid[:-1], cost_plot, '-', color='tab:blue')
    axes.fill_between(tgrid[:-1], cost_plot, color='tab:blue')
    if final_term is not None:
        legend = legend + ['final term']
        axes.arrow(tgrid[-2], 0, 0, final_cost, color='tab:pink')
    axes.legend(legend)
    # Displaying the numerical value
    string = f'Cost Func: {cumulated_cost[-1]:.2e} \nFinal Term: {final_cost:.2e}'
    axes.text(0,1.015,string, transform = axes.transAxes)
    # Setting the labels
    axes.set_xlabel('time')
    axes.set_ylabel('cost function')
    axes.grid()
    return fig

def plot_constraints(q, qd, qdd, ee_pos, u, constraints, tgrid):
    n_constr = len(constraints)
    n_joints = len(q)

    figures = []

    # Refactoring functions
    ref_joint = lambda q : [np.array([q[j][idx] for j in range(n_joints)]) for idx in range(len(q[0]))]
    ref_constr = lambda v: [np.array([array[i] for array in v]) for i in range(length)]

    for idx, constraint in enumerate(constraints):
        
        # Calculating the value of the contraints all along the x axis
        iterator = zip(ref_joint(q), ref_joint(qd), ref_joint(qdd), ref_joint(ee_pos), ref_joint(u))
        constr_plot = [constraint(q,qd,qdd,ee_pos,u) for q,qd,qdd,ee_pos,u in iterator]

        # Unpacking value and bounds
        low_bound = np.array([instant[0] for instant in constr_plot])
        value = np.array([instant[1] for instant in constr_plot])
        high_bound = np.array([instant[2] for instant in constr_plot])
        # trasform to array internal elements if necessary
        if not isinstance(low_bound[0], np.ndarray): low_bound = np.array([[el] for el in low_bound])
        if not isinstance(value[0], np.ndarray): value = np.array([[el] for el in value])
        if not isinstance(high_bound[0], np.ndarray): high_bound = np.array([[el] for el in high_bound])

        # Creating the plot
        length = len(value[0]) # colud be n_joints (if cnstraint is array) or T (in constr is scalar)
        fig, axes = plt.subplots(nrows=1, ncols=length, figsize=(img_width,3))

        if length == 1: axes = [axes]
        iterator = zip(ref_constr(low_bound), ref_constr(value), ref_constr(high_bound), axes)

        # Iterate trough each constraint along the time
        for n, (lb, val, ub, ax) in enumerate(iterator):
            ax.set_facecolor((1.0, 0.45, 0.4))

            # Painting the ok zone in white
            ax.fill_between(tgrid[:-1], lb, ub, color='w')
            ax.plot(tgrid[:-1], lb, '-', color='tab:red')
            ax.plot(tgrid[:-1], ub, '-', color='tab:red')
            ax.plot(tgrid[:-1], val, '-')
            ax.set_xlim(tgrid[0], tgrid[-2])
            ax.set_xlabel('time')
            ax.set_title('Constraint '+str(idx)+ ' ' +str([n]))
            ax.grid()
        fig.suptitle('Constraints n.' + str(idx), fontsize=14)
        fig.tight_layout()
        figures.append(fig)
    return figures

def eval_final_constr(q, qd, qdd, ee_pos, u, fconstr):
    # Function that returns the value of the last timestep
    get_last = lambda x: np.array([x[i][-1] for i in range(len(q))])
    # Retrieving the values at last timestep
    qf = get_last(q)
    qdf = get_last(qd)
    qddf = get_last(qdd)
    ee_posf = ee_pos[-1]
    uf = get_last(u)
    # Computing the final constraints values
    results = []
    if fconstr:
        for fcon in fconstr:
            results.append(fcon(qf,qdf,qddf,ee_posf,uf))
    # Results are formatted as a list of [lower bound, actual value, upper bound]
    return results

def generate_html(figure1_, figure2_, figure3_, final_constraints):
    template_path = pathlib.Path(__file__).parent.absolute()
    # Template handling
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(searchpath=template_path))
    template = env.get_template('template.html')
    img1 = encode_figure(figure1_)
    img2 = encode_figure(figure2_)
    img3 = encode_figure(figure3_)
    html = template.render(my_figure1=img1, my_figure2=img2, my_figure3=img3, final_res = final_constraints)
    
    # Write the HTML file
    name, _ = sys.argv[0].split('.', 1)
    with open(name + '_report.html', 'w') as f:
        f.write(html)
    webbrowser.open_new(name + '_report.html')

def encode_figure(figure_list):
    if not isinstance(figure_list, list): figure_list = [figure_list]
    figure_html = ''
    for fig in figure_list:
        tmpfile = BytesIO()
        fig.savefig(tmpfile, format='png')
        encoded = base64.b64encode(tmpfile.getvalue()).decode('utf8')
        figure_html = figure_html + '<img src=\'data:image/png;base64,{}\'>'.format(encoded)
        if not fig == figure_list[-1]: # if not last element
            figure_html = figure_html + '<br>' # line break
    return figure_html