all:
	$(MAKE) -C RaycastWorld
	$(MAKE) -C Game
	$(MAKE) -C PycastWorld

clean:
	$(MAKE) -C RaycastWorld clean
	$(MAKE) -C Game clean
	$(MAKE) -C PycastWorld clean
