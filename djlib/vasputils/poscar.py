import numpy as np
import os

# Defines a poscar class and functions to manipulate poscars from vasp.


class poscar:
    def __init__(self, poscar_file_path):
        self.poscar_file_path = poscar_file_path
        self.pos_name = ""
        self.species_vec = []
        self.species_count = np.array([])
        self.basis_scaling = 1
        self.basis = np.zeros((3, 3))
        self.coord_style = "Direct"  # in vasp, direct == fractional
        self.coords = []

        lineCount = 0
        readCoords = True
        coord_line = 7
        special_settings = []
        with open(self.poscar_file_path, "r") as pfile:
            for line in pfile:

                if len(line.split()) == 0:
                    # print('ERROR: unexpected empty line at line %s\nScript might not work properly.' % (lineCount +1) )
                    # print("(if this is a CONTCAR and the problem line is after the coordinates, you're fine)\n\n")
                    readCoords = False

                if lineCount == 0:
                    self.pos_name = line
                elif lineCount == 1:
                    self.basis_scaling = float(line)
                elif lineCount > 1 and lineCount < 5:
                    self.basis[lineCount - 2, :] = np.array(line.split()).astype(float)
                elif lineCount == 5:
                    self.species_vec = line.split()
                elif lineCount == 6:
                    self.species_count = np.array(line.split()).astype(int)
                elif lineCount == 7:
                    if line.split()[0][0] == "d" or line.split()[0][0] == "D":
                        self.coord_style = "Direct"
                    elif line.split()[0][0] == "c" or line.split()[0][0] == "C":
                        self.coord_style = "Cartesian"
                    else:
                        special_settings.append(line.strip())
                        coord_line = coord_line + 1

                elif lineCount > coord_line and readCoords:
                    self.coords.append(
                        line.split()[0:3]
                    )  # will chop of any descriptors
                lineCount += 1

        pfile.close()
        self.coords = np.array(self.coords).astype(float)

    def writePoscar(self):
        # writes the poscar to a file
        currentDirectory = ""
        for i in range(len(self.poscar_file_path.split("/")) - 1):
            currentDirectory = (
                currentDirectory + "/" + self.poscar_file_path.split("/")[i]
            )
        currentDirectory = currentDirectory[1:]

        with open(os.path.join(currentDirectory, "newPoscar.vasp"), "w") as newPoscar:
            newPoscar.write("new_poscar_" + self.pos_name)
            newPoscar.write("%f\n" % self.basis_scaling)

            for row in self.basis:
                for element in row:
                    newPoscar.write(str(element) + " ")
                newPoscar.write("\n")

            for species in self.species_vec:
                newPoscar.write(species + " ")
            newPoscar.write("\n")

            for count in self.species_count:
                newPoscar.write(str(count) + " ")
            newPoscar.write("\n")

            newPoscar.write("%s\n" % self.coord_style)

            for row in self.coords:
                if True:  # all(row < 1):
                    for element in row:
                        newPoscar.write(str(element) + " ")
                    newPoscar.write("\n")
        newPoscar.close()
