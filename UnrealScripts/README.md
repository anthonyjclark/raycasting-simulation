# Creating Maze Project

1. Open UE4.26.x
2. Create new project (blank, blueprint, no starter, etc.)
3. Enable plugins (Editor -> Plugins)
    - Editor Scripting Utilities
    - Python Editor Script Plugin
    - UnrealCV Plugin
4. Create the actors. Drag the Cube to the "Content Browser" five times.
	- Floor
	- Wall
	- WallLeft
	- WallRight
	- WallGoal
5. Drag textures into "Content Browser"
6. Create material from each texture (right click)
7. Fix floor
	- Open material
	- Add TexCoord
	- Connect to UVs
8. Double click on Actors and apply materials
9. Add camera


# Building and Installing UnrealCV Plugin On Linux

1. Fix "Pending" bug by changing the name.
2. Build UE4: `make`
3. Build the plugin: `python build.py --install --UE4 /home/ajc/UE4/UnrealEngine/`
4. Enable the plugin in the project
