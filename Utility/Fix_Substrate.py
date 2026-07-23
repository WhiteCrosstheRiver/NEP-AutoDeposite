def process_xyz_file(input_file, output_file, fix_regions):
    """
    Reads a non-standard XYZ file, converts it to the GPUMD format with group labels,
    and writes the converted data to a new file.

    Parameters:
    - input_file (str): Path to the input XYZ file.
    - output_file (str): Path to the output GPUMD XYZ file.
    - fix_regions (list of lists): List of regions, each defined as [[xmin, xmax], [ymin, ymax], [zmin, zmax]].
    """
    with open(input_file, 'r') as infile:
        lines = infile.readlines()

    # Read the number of atoms and lattice info from the header
    num_atoms = int(lines[0].strip())
    header_line = lines[1].strip()

    # Prepare the new header line in GPUMD format
    if "group:I:1" not in header_line:
        header_line += " properties=species:S:1:pos:R:3:group:I:1"

    converted_lines = [str(num_atoms), header_line]

    # Process each atom line
    for line in lines[2:]:
        components = line.strip().split()
        species = components[0]
        x, y, z = map(float, components[1:4])

        # Initialize group label to 0 (not fixed)
        group_label = 0

        # Check if the atom falls within any of the specified fix regions
        for region in fix_regions:
            [xmin, xmax], [ymin, ymax], [zmin, zmax] = region
            if xmin <= x <= xmax and ymin <= y <= ymax and zmin <= z <= zmax:
                group_label = 1  # Mark the atom as fixed in the specified region
                break  # Exit loop if atom is in any of the regions

        # Add the updated atom line with group label
        converted_line = f"{species} {x} {y} {z} {group_label}"
        converted_lines.append(converted_line)

    # Write the converted data to the output file
    with open(output_file, 'w') as outfile:
        for line in converted_lines:
            outfile.write(line + "\n")

    print(f"Conversion complete. The output is saved to {output_file}.")

# Example usage
test_input_file = '111-.xyz'
test_output_file = '111.xyz'

fix_bottom = [
    [[0.0, 999], [0.0, 999.0], [0.0, 3.0]]  # fix bottom side edge
]

process_xyz_file(test_input_file, test_output_file, fix_bottom)
