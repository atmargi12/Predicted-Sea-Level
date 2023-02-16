from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pandas as pd
import PIL, PIL.Image

import scipy.stats as st
import scipy.interpolate

def calculate_std(upper, mean):
    """
	Calculate standard deviation based on the upper 95th percentile

	Args:
		upper: a 1-d numpy array with length N, representing the 95th percentile
            values from N data points
		mean: a 1-d numpy array with length N, representing the mean values from
            the corresponding N data points

	Returns:
		a 1-d numpy array of length N, with the standard deviation corresponding
        to each value in upper and mean
	"""
    return (upper - mean) / st.norm.ppf(.975)


def interp(target_year, input_years, years_data):
    """
	Interpolates data for a given year, based on the data for the years around it

	Args:
		target_year: an integer representing the year which you want the predicted
            sea level rise for
		input_years: a 1-d numpy array that contains the years for which there is data
		    (can be thought of as the "x-coordinates" of data points)
        years_data: a 1-d numpy array representing the current data values
            for the points which you want to interpolate, eg. the SLR mean per year data points
            (can be thought of as the "y-coordinates" of data points)

	Returns:
		the interpolated predicted value for the target year
	"""
    return np.interp(target_year, input_years, years_data, right=-99)


def load_data():
    """
	Loads data from sea_level_change.csv and puts it into numpy arrays

	Returns:
		a length 3 tuple of 1-d numpy arrays:
		    1. an array of years as ints
		    2. an array of 2.5th percentile sea level rises (as floats) for the years from the first array
		    3. an array of 97.5th percentile of sea level rises (as floats) for the years from the first array
        eg.
            (
                [2020, 2030, ..., 2100],
                [3.9, 4.1, ..., 5.4],
                [4.4, 4.8, ..., 10]
            )
            can be interpreted as:
                for the year 2020, the 2.5th percentile SLR is 3.9ft, and the 97.5th percentile would be 4.4ft.
	"""
    df = pd.read_csv('sea_level_change.csv')
    df.columns = ['Year', 'Lower', 'Upper']
    return (df.Year.to_numpy(), df.Lower.to_numpy(), df.Upper.to_numpy())


def predicted_sea_level_rise(show_plot=False):
    """
	Creates a numpy array from the data in sea_level_change.csv where each row
    contains a year, the mean sea level rise for that year, the 2.5th percentile
    sea level rise for that year, the 97.5th percentile sea level rise for that
    year, and the standard deviation of the sea level rise for that year. If
    the year is between 2020 and 2100 and not included in the data, the values
    for that year should be interpolated. If show_plot, displays a plot with
    mean and the 95%, assuming sea level rise follows a linear trend.

	Args:
		show_plot: displays desired plot if true

	Returns:
		a 2-d numpy array with each row containing a year in order from 2020-2100
        inclusive, mean, the 2.5th percentile, 97.5th percentile, and standard
        deviation of the sea level rise for the given year
	"""
    # Load our data
    d = load_data()
    
    # Create an array with 81 rows and 5 columns
    ans = np.zeros((81, 5))
    
    # For the years that exist, generate its respective row. Use the mean
    # of upper and lower to find the mean, and calculate std using the helper
    # function, with parameters upper and mean.
    for i in range(len(d[0])):
        e = d[0][i]
        ans[e - 2020][0] = e
        ans[e - 2020][1] = (d[1][i] + d[2][i]) / 2
        ans[e - 2020][2] = d[1][i]
        ans[e - 2020][3] = d[2][i]
        ans[e - 2020][4] = calculate_std(d[2][i], ans[e-2020][1])
        
    # For the years not already existing, interpolate the upper and lower
    # using the already existing data. Then, populate ans like previously
    # using this new data.    
        
    for i in range(0, 81):
        if i + 2020 not in d[0]:
            ans[i][0] = i + 2020
            e1 = interp(i + 2020, d[0], d[1])
            e2 = interp(i + 2020, d[0], d[2])
            ans[i][1] = (e1 + e2) / 2
            ans[i][2] = e1
            ans[i][3] = e2
            ans[i][4] = calculate_std(e2, ans[i][1])
    
    # Plot the three sea levels separately.
    
    if show_plot:
        plt.plot(ans[:, 0], ans[:, 2], label = 'Lower', linestyle = "--")
        plt.plot(ans[:, 0], ans[:, 1], label = 'Mean')
        plt.plot(ans[:, 0], ans[:, 3], label = 'Upper', linestyle = "--")
        plt.xlabel('Year')
        plt.ylabel('Projected annual mean water level (ft)')
        plt.show()
    return ans


def simulate_year(data, year, num):
    """
	Simulates the sea level rise for a particular year based on that year's
    mean and standard deviation, assuming a normal distribution.

	Args:
		data: a 2-d numpy array with each row containing a year in order from 2020-2100
            inclusive, mean, the 2.5th percentile, 97.5th percentile, and standard
            deviation of the sea level rise for the given year
		year: the year to simulate sea level rise for
        num: the number of samples you want from this year

	Returns:
		a 1-d numpy array of length num, that contains num simulated values for
        sea level rise during the year specified
	"""
    
    # The index of the year is year - 2020.
    
    index = year - 2020
    
    # Recover the mean and the std using the index.
    
    mean = data[index][1]
    std = data[index][4]
    
    # Return num normal random estimated using mean and std.
    
    return np.random.normal(mean, std, num)
    
def plot_simulation(data):
    """
	Runs and plots a Monte Carlo simulation, based on the values in data and
    assuming a normal distribution. Five hundred samples should be generated
    for each year.

	Args:
		data: a 2-d numpy array with each row containing a year in order from 2020-2100
            inclusive, mean, the 2.5th percentile, 97.5th percentile, and standard
            deviation of the sea level rise for the given year
	"""
    
    # Initialize a result array.
    
    result = np.zeros(81) 
    
    L = []
    
    
    # For each trial, generate a list of simulated values for each year. Then,
    # scatter plot this simulated data for each year.
    
    for i in range(500):
        L = []
        for j in range(81):
            L.append(simulate_year(data, j + 2020, 1))
        plt.scatter(data[:, 0], L, c = 'gray', s = 2, alpha = 0.3)
        
    # Finally, plot the mean, lower and upper.    
    
    plt.plot(data[:, 0], data[:, 2], label = 'Lower', linestyle = "--")
    plt.plot(data[:, 0], data[:, 1], label = 'Mean')
    plt.plot(data[:, 0], data[:, 3], label = 'Upper', linestyle = "--")
    plt.xlabel('Year')
    plt.ylabel('Relative Water Level Change (ft)')
    plt.show()

def simulate_water_levels(data):
    """
	Simulates the water level for all years in the range 2020 to 2100, inclusive.

	Args:
		data: a 2-d numpy array with each row containing a year in order from 2020-2100
            inclusive, mean, the 2.5th percentile, 97.5th percentile, and standard
            deviation of the sea level rise for the given year

	Returns:
		a list of simulated water levels for each year, in the order in which
        they would occur temporally
	"""
    
    # For each year, store a simulated value in L, and return the list.
    
    L = []
    for i in range(81):
        L.append(simulate_year(data, i + 2020, 1)[0])
    return L


def repair_only(water_level_list, water_level_loss_no_prevention, house_value=400000):
    """
	Simulates the water level for all years in the range 2020 to 2100, inclusive,
    and calculates damage costs in 1000s resulting from a particular water level
    for each year dependent on a repair only strategy, where you would only pay
    to repair damage that already happened.

    The specific damage cost can be calculated using the numpy array
    water_level_loss_no_prevention, where each water level corresponds to the
    percent of property that is damaged.

    The repair only strategy is as follows:
        1) If the water level is less than or equal to 5ft, the cost is 0.
        2) If the water level is between 5ft and 10ft, the cost is the
           house_value times the percentage of property damage for that water
           level. If the water level is not an integer value, the percentage
           should be interpolated.
        3) If the water level is at least 10ft, the cost is the entire value of
           the house.

	Args:
		water_level_list: list of simulated water levels for 2020-2100
        water_level_loss_no_prevention: a 2-d numpy array where the first column is
            the SLR levels and the second column is the corresponding property damage expected
            from that water level with no flood prevention (as an integer percentage)
        house_value: the value of the property we are estimating cost for

	Returns:
		an list of damage costs in 1000s, in the order in which the costs would
        be incurred temporally
	"""
    
    # For all the values in the water list, compare them with 5 and 10.
    # If lower than 5, attend 0, if greater than 10, append 400, and if
    # in between, check if the water level is an integer. If so, just
    # append the percentage in water_level_no_prevention times 4. If not,
    # just append the interpolation times 4.
    
    n = len(water_level_list)
    L = []
    for i in range(n):
        if water_level_list[i] < 5:
            L.append(0)
        if water_level_list[i] >= 10:
            L.append(400)
        elif water_level_list[i] > 5 and water_level_list[i] < 10:
            if water_level_list[i].is_integer():
                L.append(water_level_loss_no_prevention[water_level_list[i] - 5, 1] * 4)
            else:
                L.append(scipy.interpolate.interp1d(water_level_loss_no_prevention[:, 0], water_level_loss_no_prevention[:, 1], fill_value = 'extrapolate')(water_level_list[i]) * 4)
                
    # Finally, return L.
    
    return L


def wait_a_bit(water_level_list, water_level_loss_no_prevention, water_level_loss_with_prevention, house_value=400000,
               cost_threshold=100000):
    """
	Simulates the water level for all years in the range 2020 to 2100, inclusive,
    and calculates damage costs in 1000s resulting from a particular water level
    for each year dependent on a wait a bit to repair strategy, where you start
    flood prevention measures after having a year with an excessive amount of
    damage cost.

    The specific damage cost can be calculated using the numpy array
    water_level_loss_no_prevention and water_level_loss_with_prevention, where
    each water level corresponds to the percent of property that is damaged.
    You should be using water_level_loss_no_prevention when no flood prevention
    measures are in place, and water_level_loss_with_prevention when there are
    flood prevention measures in place.

    Flood prevention measures are put into place if you have any year with a
    damage cost above the cost_threshold.

    The wait a bit to repair only strategy is as follows:
        1) If the water level is less than or equal to 5ft, the cost is 0.
        2) If the water level is between 5ft and 10ft, the cost is the
           house_value times the percentage of property damage for that water
           level, which is affected by the implementation of flood prevention
           measures. If the water level is not an integer value, the percentage
           should be interpolated.
        3) If the water level is at least 10ft, the cost is the entire value of
           the house.

	Args:
		water_level_list: list of simulated water levels for 2020-2100
        water_level_loss_no_prevention: a 2-d numpy array where the columns are
            water levels and the corresponding percent of property damage expected
            from that water level with no flood prevention
        water_level_loss_with_prevention: a 2-d numpy array where the columns are
            water levels and the corresponding percent of property damage expected
            from that water level with flood prevention
        house_value: the value of the property we are estimating cost for
        cost_threshold: the amount of cost incurred before flood prevention
            measures are put into place

	Returns:
		an list of damage costs in 1000s, in the order in which the costs would
        be incurred temporally
	"""
    
    # For all the values in the water list, compare them with 5 and 10.
    # If lower than 5, attend 0, if greater than 10, append 400, and
    # set the counter to 1, as going forward we'll use the prevention
    # plan, and if  in between, check if the water level is an integer.
    # If so, if the counter is still 0 and we have not passed the cost
    # threshold, jist append the no prevention value times 4. If the counter
    # is at 0 and we pass the threshold, append the no prevention value times
    # 4 and set the counter to 1. If the counter is 1, append the prevention
    # plan value times 4. If it is not an integer, do the same thing as for
    # integer values, but just using the interpolation values (keep track of
    # the counter in the same exact way).
    
    n = len(water_level_list)
    L = []
    c = 0
    for i in range(n):
        if water_level_list[i] <= 5:
            L.append(0)
        if water_level_list[i] >= 10:
            L.append(400)
            c = 1
        elif water_level_list[i] > 5 and water_level_list[i] < 10:
            if water_level_list[i].is_integer():
                if water_level_loss_no_prevention[water_level_list[i] - 5, 1] < cost_threshold / house_value * 100 and c == 0:
                    L.append(water_level_loss_no_prevention[water_level_list[i] - 5, 1] * 4)
                elif water_level_loss_no_prevention[water_level_list[i] - 5, 1] >= cost_threshold / house_value * 100 and c == 0:
                    L.append(water_level_loss_no_prevention[water_level_list[i] - 5, 1] * 4)
                    c = 1
                else:
                    L.append(water_level_loss_with_prevention[water_level_list[i] - 5, 1] * 4)
            else:
                if scipy.interpolate.interp1d(water_level_loss_no_prevention[:, 0], water_level_loss_no_prevention[:, 1], fill_value = 'extrapolate')(water_level_list[i]) < cost_threshold / house_value * 100 and c == 0:
                    L.append(scipy.interpolate.interp1d(water_level_loss_no_prevention[:, 0], water_level_loss_no_prevention[:, 1], fill_value = 'extrapolate')(water_level_list[i]) * 4)
                elif scipy.interpolate.interp1d(water_level_loss_no_prevention[:, 0], water_level_loss_no_prevention[:, 1], fill_value = 'extrapolate')(water_level_list[i]) >= cost_threshold / house_value * 100 and c == 0:
                    L.append(scipy.interpolate.interp1d(water_level_loss_no_prevention[:, 0], water_level_loss_no_prevention[:, 1], fill_value = 'extrapolate')(water_level_list[i]) * 4)
                    c = 1
                else:
                    L.append(scipy.interpolate.interp1d(water_level_loss_with_prevention[:, 0], water_level_loss_with_prevention[:, 1], fill_value = 'extrapolate')(water_level_list[i]) * 4)
    
    # Finally, return the list.
    
    return L



def prepare_immediately(water_level_list, water_level_loss_with_prevention, house_value=400000):
    """
	Simulates the water level for all years in the range 2020 to 2100, inclusive,
    and calculates damage costs in 1000s resulting from a particular water level
    for each year dependent on a prepare immediately strategy, where you start
    flood prevention measures immediately.

    The specific damage cost can be calculated using the numpy array
    water_level_loss_with_prevention, where each water level corresponds to the
    percent of property that is damaged.

    The prepare immediately strategy is as follows:
        1) If the water level is less than or equal to 5ft, the cost is 0.
        2) If the water level is between 5ft and 10ft, the cost is the
           house_value times the percentage of property damage for that water
           level, which is affected by the implementation of flood prevention
           measures. If the water level is not an integer value, the percentage
           should be interpolated.
        3) If the water level is at least 10ft, the cost is the entire value of
           the house.

	Args:
		water_level_list: list of simulated water levels for 2020-2100
        water_level_loss_with_prevention: a 2-d numpy array where the columns are
            water levels and the corresponding percent of property damage expected
            from that water level with flood prevention
        house_value: the value of the property we are estimating cost for

	Returns:
		an list of damage costs in 1000s, in the order in which the costs would
        be incurred temporally
	"""
    
    # Do the same things as for repair only, but just use the water level
    # with prevention directly here instead of the no prevention.
    
    n = len(water_level_list)
    L = []
    for i in range(n):
        if water_level_list[i] < 5:
            L.append(0)
        if water_level_list[i] >= 10:
            L.append(400)
        elif water_level_list[i] > 5 and water_level_list[i] < 10:
            if water_level_list[i].is_integer():
                L.append(water_level_loss_with_prevention[water_level_list[i] - 5, 1] * 4)
            else:
                L.append(scipy.interpolate.interp1d(water_level_loss_with_prevention[:, 0], water_level_loss_with_prevention[:, 1], fill_value = 'extrapolate')(water_level_list[i]) * 4)
    
    # Return the list.
    
    return L


def plot_strategies(data, water_level_loss_no_prevention, water_level_loss_with_prevention, house_value=400000,
                    cost_threshold=100000):
    """
	Runs and plots a Monte Carlo simulation of all of the different preparation
    strategies, based on the values in data and assuming a normal distribution.
    Five hundred samples should be generated for each year.

	Args:
		data: a 2-d numpy array with each row containing a year in order from 2020-2100
            inclusive, the 5th percentile, 95th percentile, mean, and standard
            deviation of the sea level rise for the given year
        water_level_loss_no_prevention: a 2-d numpy array where the columns are
            water levels and the corresponding percent of property damage expected
            from that water level with no flood prevention
        water_level_loss_with_prevention: a 2-d numpy array where the columns are
            water levels and the corresponding percent of property damage expected
            from that water level with flood prevention
        house_value: the value of the property we are estimating cost for
        cost_threshold: the amount of cost incurred before flood prevention
            measures are put into place
	"""
    
    # Initialize lists which will contain repair only, wait a bit and
    # prepare immediately values, to be used later.
    
    x = []
    y = []
    z = []
    
    
    # For each trial, generate a simulated water level list, and find
    # the repair only, wait a bit and prepare immediately lists for each.
    # Append these lists to our master lists, and then plot the data
    # found for the free preparation types.
    
    for i in range(500):
        water_level_list = simulate_water_levels(data)
        a = repair_only(water_level_list, water_level_loss_no_prevention)
        b = wait_a_bit(water_level_list, water_level_loss_no_prevention, water_level_loss_with_prevention)
        c = prepare_immediately(water_level_list, water_level_loss_with_prevention)
        x.append(a)
        y.append(b)
        z.append(c)
        
        plt.scatter(data[:, 0], a, c = 'gray', s = 2, alpha = 0.1)
        plt.scatter(data[:, 0], b, c = 'gray', s = 2, alpha = 0.1)
        plt.scatter(data[:, 0], c, c = 'gray', s = 2, alpha = 0.1)
        
    # Finally, compute the means for each preparation type, and plot them
    # for each year. Finally, show the plot.
        
    repair = np.mean(x, axis = 0)
    wait = np.mean(y, axis = 0)
    immed = np.mean(z, axis = 0)
    
    plt.plot(data[:, 0], repair, label = 'Repair-only scenario', c = 'green')
    plt.plot(data[:, 0], wait, label = 'Wait-a-bit scenario', c = 'blue')
    plt.plot(data[:, 0], immed, label = 'Prepare-immediately scenario', c = 'red')
    plt.xlabel('Year')
    plt.ylabel('Estimated Damage Cost ($K)')
    plt.show()


if __name__ == '__main__':
    
    # Comment out the 'pass' statement below to run the lines below it
    pass 

    # # Uncomment the following lines to plot generate plots
    data = predicted_sea_level_rise()
    water_level_loss_no_prevention = np.array([[5, 6, 7, 8, 9, 10], [0, 10, 25, 45, 75, 100]]).T
    water_level_loss_with_prevention = np.array([[5, 6, 7, 8, 9, 10], [0, 5, 15, 30, 70, 100]]).T
    plot_simulation(data)
    plot_strategies(data, water_level_loss_no_prevention, water_level_loss_with_prevention)