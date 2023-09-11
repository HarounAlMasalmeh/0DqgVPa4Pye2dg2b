import matplotlib.pyplot as plt


def visualize(train_generator):
    x_batch, y_batch = next(train_generator)

    # Define the number of images to display
    num_images_to_display = 16

    # Create a grid of subplots
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))

    # Loop over the axes to plot each image
    for i, ax in enumerate(axes.flat):
        # Display an image at the i-th position
        ax.imshow(x_batch[i])

        # Display the label as the title
        ax.set_title(f'Label: {y_batch[i]}')

        # Hide grid lines
        ax.grid(False)

        # Hide axes ticks
        ax.set_xticks([])
        ax.set_yticks([])

    # Adjust layout so plots are not overlapping
    plt.tight_layout()

# Show the plot
plt.show()