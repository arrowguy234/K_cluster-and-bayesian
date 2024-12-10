import pandas as pd  # Importing pandas for data manipulation and analysis
import matplotlib.pyplot as plt  # Importing matplotlib for plotting

# Read the Gaia data from a CSV file into a pandas DataFrame
data = pd.read_csv('gaia_data.csv')

# Plotting the sky distribution: Right Ascension (RA) vs Declination (Dec)
plt.figure(figsize=(8, 6))  # Set the figure size for the plot
plt.scatter(data['ra'], data['dec'], alpha=0.5, s=1)  # Scatter plot with RA on the x-axis and Dec on the y-axis
plt.title('Sky Distribution: RA vs Dec')  # Title for the plot
plt.xlabel('Right Ascension (RA)')  # Label for the x-axis
plt.ylabel('Declination (Dec)')  # Label for the y-axis
plt.grid(True)  # Enable grid for easier visualization of the data points
plt.show()  # Display the plot

# Plotting the distribution of parallax
plt.figure(figsize=(8, 6))  # Set the figure size for the plot
plt.hist(data['parallax'], bins=50, color='skyblue', edgecolor='black')  # Histogram of parallax with 50 bins, skyblue color, and black edges
plt.title('Distribution of Parallax')  # Title for the plot
plt.xlabel('Parallax (mas)')  # Label for the x-axis
plt.ylabel('Frequency')  # Label for the y-axis
plt.grid(True)  # Enable grid for better visualization
plt.show()  # Display the plot

# Plotting the distribution of G Magnitude
plt.figure(figsize=(8, 6))  # Set the figure size for the plot
plt.hist(data['phot_g_mean_mag'], bins=50, color='orange', edgecolor='black')  # Histogram of G Magnitude with 50 bins, orange color, and black edges
plt.title('Distribution of G Magnitude')  # Title for the plot
plt.xlabel('G Magnitude')  # Label for the x-axis
plt.ylabel('Frequency')  # Label for the y-axis
plt.grid(True)  # Enable grid for easier visualization
plt.show()  # Display the plot

# Plotting the distribution of star distances
plt.figure(figsize=(8, 6))  # Set the figure size for the plot
plt.hist(data['distance_gspphot'], bins=50, color='green', edgecolor='black')  # Histogram of star distances with 50 bins, green color, and black edges
plt.title('Distribution of Star Distances')  # Title for the plot
plt.xlabel('Distance (pc)')  # Label for the x-axis
plt.ylabel('Frequency')  # Label for the y-axis
plt.grid(True)  # Enable grid for better visualization
plt.show()  # Display the plot

# Plotting the density of stars: RA vs Dec using hexbin
plt.figure(figsize=(8, 6))  # Set the figure size for the plot
plt.hexbin(data['ra'], data['dec'], gridsize=50, cmap='YlGnBu')  # Create a hexbin plot with RA and Dec, 50 hexagonal bins, and a color map
plt.colorbar(label='Density')  # Add a color bar to represent the density of stars in the hexbin plot
plt.title('Density of Stars (RA vs Dec)')  # Title for the plot
plt.xlabel('Right Ascension (RA)')  # Label for the x-axis
plt.ylabel('Declination (Dec)')  # Label for the y-axis
plt.show()  # Display the plot

# Scatter plot of RA vs Dec colored by distance for each star
plt.figure(figsize=(10, 8))  # Set the figure size for the plot
scatter = plt.scatter(data['ra'], data['dec'], c=data['distance_gspphot'], cmap='viridis', s=40, alpha=0.6, edgecolors='w')  # Scatter plot with RA on the x-axis, Dec on the y-axis, and color representing distance
plt.colorbar(scatter, label='Distance (pc)')  # Add a color bar to show the distance of each star in parsecs
plt.xlabel('Right Ascension (ra) in degrees')  # Label for the x-axis
plt.ylabel('Declination (dec) in degrees')  # Label for the y-axis
plt.title('RA vs DEC Colored by Distance for Each Star')  # Title for the plot
plt.show()  # Display the plot
