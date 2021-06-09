# Creating Maze Project

1. Open UE4.26.x
2. Create new project (blank, blueprint, no starter, etc.)
3. Enable plugins (Editor -> Plugins)
    - Editor Scripting Utilities
    - Python Editor Script Plugin
    - UnrealCV Plugin (HamiltonHuaji/unrealcv)
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
9. Set Maze level as default (Edit => Project Settings -> Maps & Modes -> Game Default Map)
10. Create `./Config/DefaultGameUserSettings.ini` with:

```ini
[/Script/Engine.GameUserSettings]
ResolutionSizeX=320
ResolutionSizeY=240
LastUserConfirmedResolutionSizeX=320
LastUserConfirmedResolutionSizeY=240
WindowPosX=-1
WindowPosY=-1
bUseDesktopResolutionForFullscreen=False
FullscreenMode=2
LastConfirmedFullscreenMode=2
Version=5
```
11. Package the game (File->Package Project->[Platform])

# Building and Installing UnrealCV Plugin On Linux

1. Fix "Pending" bug by changing the name.
2. Build UE4: `make`
3. Build the plugin: `python build.py --install --UE4 /home/ajc/UE4/UnrealEngine/`
4. Enable the plugin in the project

## Manual Package Install

1. Clone the working UnrealCV reporitory
2. cd into unrealCV/client/python/
3. run `python setup.py install`

# Unreal Tasks

- Clean up ImitateInUnreal notebook i.e. add documentation and make it easy to interact with code 
- Add multiple levels to the Unreal Project; generate 20 mazes under the mazes directory
	- generate them in unreal 
	- look at `vset /action/game/level [level_name]`
- Cycle-GAN: 
	- create paired images(exact same coordinates and pose) from raycasting and unreal 
