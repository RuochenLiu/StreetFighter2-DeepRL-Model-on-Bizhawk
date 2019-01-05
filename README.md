# Reinforcement Learning for Street Fighter 2
## Overview
For this project, I set up an online DRL training environment for Street Fighter 2 (Sega MD) on Bizhawk and with the same method, we could train models for any other games on Bizhawk.  
There are two parts of code. A Python script with socket server and neural network will receive observations, return actions, and update weights of NN. A Lua script with socket client will grab, preprocess and send observations, take actions and control the emulator.
### Testing preview
![](https://github.com/RuochenLiu/StreetFighter2-DeepRL-Model-on-Bizhawk/blob/master/image/vsGuile_slow.gif)  ![](https://github.com/RuochenLiu/StreetFighter2-DeepRL-Model-on-Bizhawk/blob/master/image/vsFeilong_slow.gif)
## Bizhawk and related work
‘[BizHawk](http://tasvideos.org/BizHawk.html) is a multi-system emulator, which provides nice features for casual gamers such as fullscreen, rewind, and joypad support in addition to rerecording and debugging tools for all system cores.’
With these great features, there are already some great ML projects trained on Bizhawk, such as [Mario I/O](https://pastebin.com/ZZmSNaHX) which is a NEAT model trained for first level of Super Mario, and [Neural Kart](https://github.com/rameshvarun/NeuralKart) which is trained offline and tested online for Mario Kart.
## Environment
### State
Screenshot at each frame is the first part of observation, which contains lots of info and can be sent to clipboard by Lua script for Python training. RAM value vectors are used as the second part of observation. RAM search and RAM watch tools of Bizhawk helped a lot, which let you find the memory address of a specific move. Here is a great [tutorial](https://www.youtube.com/watch?v=zsPLCIAJE5o&t=2064s) by The8bitbeast.
```lua
local p1HPAddr = 0x803C
local p2HPAddr = 0x833C
local p1XAddr = 0x8006
local p2XAddr = 0x83D8
local yDiffAddr = 0x80CE
local timeRemainAddr = 0x986E
local p1CharacterAddr = 0x828A
local p2CharacterAddr = 0x858A
local p1InActAddr = 0x80BE
local p2InActAddr = 0x83BE
local p1CrouchAddr = 0x8054
local p2CrouchAddr = 0x8354
```
### Action Space
I set Sagat as agent for this project, and given the system of SF2 is not that complicated as USF4, I just chose basic moves, special moves and block as the whole action space. There are no combo moves pre-set.
```lua
-- Crouch (Attack)
function myMove.crouch(button)
    if button == nil then
        myInput({"Down"})
    else
        myInput({"Down", button})
    end
end
-- Jump (Attack)
function myMove.jump(button)
    if button == nil then
        myInput({"Up"})
    else
        myInput({"Up", button})
    end
end
```
```lua
-- Tiger 236P/K
function myMove.tiger(direction, button)
    if direction < 0 then
        myInput({"Down"})
        myInput({"Down", "Right"})
        myInput({"Right"})
        myInput({button})
    else
        myInput({"Down"})
        myInput({"Down", "Left"})
        myInput({"Left"})
        myInput({button})
    end
end
-- TigerUppercut 6236P
function myMove.uppercut(direction, button)
    if direction < 0 then
        myInput({"Down", "Right"})
        myInput({"Down", "Right"})
        myInput({button})
    else
        myInput({"Down", "Left"})
        myInput({"Down", "Left"})
        myInput({button})
    end
end
```
### Reward Function
Change of HP difference after each chosen action is regarded as reward and I set a bonus for each block action because reward of blocking is always negative.
### Training
Python is set as neural network server while Lua is set as agent client with socket tcp connection for training. The process flow graph is below.
![test result](https://github.com/RuochenLiu/StreetFighter2-DeepRL-Model-on-Bizhawk/blob/master/image/Deep%20RL%20for%20Street%20Fighter%202.jpg)
### Result
After 1200 episodes trained, my Sagat is tested by fighting against all players and winning rates are here.
![test result](https://github.com/RuochenLiu/StreetFighter2-DeepRL-Model-on-Bizhawk/blob/master/image/test_result.png)
Some parts still need to be fixed. Each chosen action affects the following frames and the recovery time may cause negative rewards, which has nothing to do with the next action chose by the model. Combo moves are hard to be learnt because start-up, duration and recovery time for each move differs from each other.
