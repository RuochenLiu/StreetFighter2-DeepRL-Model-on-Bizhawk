-- Author: RuochenLiu
-- Email: ruochen.liu@columbia.edu
-- Version: 1.0.0

local host, port = "127.0.0.1", 8001
local socket = require("socket")
local tcp = assert(socket.tcp())
tcp:connect(host, port)
tcp:send('Connected')

local p1HPAddr = 0x803C
local p2HPAddr = 0x833C
local p1XAddr = 0x8006
local p2XAddr = 0x83D8
local yDiffAddr = 0x80CE
local p1MovingAddr = 0x80E6
local p2MovingAddr = 0x83E6
local p1JumpAddr = 0x8444
local p2JumpAddr = 0x8144
local timeRemainAddr = 0x986E
local loseCountAddr = 0x8284
local gameStageAddr = 0x9A04
local p1CharacterAddr = 0x828A
local p2CharacterAddr = 0x858A
local p1InActAddr = 0x80BE
local p2InActAddr = 0x83BE
local p1CrouchAddr = 0x8054
local p2CrouchAddr = 0x8354
local p2FireballAddr = 0x8686
local fc = 0

-- Memory class
local getMemory = {}
function getMemory.new()
    local self = setmetatable({}, getMemory)
    self.p1HP = memory.read_u16_be(p1HPAddr, "68K RAM")
    self.p2HP = memory.read_u16_be(p2HPAddr, "68K RAM")
    self.p1X = memory.read_u16_be(p1XAddr, "68K RAM")
    self.p2X = memory.read_u16_be(p2XAddr, "68K RAM")
    self.yDiff = memory.read_u16_be(yDiffAddr, "68K RAM")
    self.p1Moving = memory.read_u16_be(p1MovingAddr, "68K RAM")
    self.p2Moving = memory.read_u16_be(p2MovingAddr, "68K RAM")
    self.p1Jump = memory.read_u16_be(p1JumpAddr, "68K RAM")
    self.p2Jump = memory.read_u16_be(p2JumpAddr, "68K RAM")
    self.timeRemain = memory.read_u16_be(timeRemainAddr, "68K RAM")
    self.loseCount = memory.read_u16_be(loseCountAddr, "68K RAM")
    self.gameStage = memory.read_u16_be(gameStageAddr, "68K RAM")
    self.p1Character = memory.read_u16_be(p1CharacterAddr, "68K RAM")
    self.p2Character = memory.read_u16_be(p2CharacterAddr, "68K RAM")
    self.p1InAct = memory.read_u16_be(p1InActAddr, "68K RAM")
    self.p2InAct = memory.read_u16_be(p2InActAddr, "68K RAM")
    self.p1Crouch = memory.read_u16_be(p1CrouchAddr, "68K RAM")
    self.p2Crouch = memory.read_u16_be(p2CrouchAddr, "68K RAM")
    return self
end
-- Input function
function myInput(trueList)
    defaultInput = {["A"] = false,
                    ["B"] = false,
                    ["C"] = false,
                    ["Down"] = false,
                    ["Left"] = false,
                    ["Mode"] = false,
                    ["Right"] = false,
                    ["Start"] = false,
                    ["Up"] = false,
                    ["X"] = false,
                    ["Y"] = false,
                    ["Z"] = false}
    n = #trueList
    for i = 1,n do
        defaultInput[trueList[i]] = true
    end
    joypad.set(defaultInput, 1)
    emu.frameadvance()
end

-- Frame-wait
function frameWait(n)
    i = 1
    while i <= n do
        emu.frameadvance()
        i = i + 1
    end 
end

-- Move class
local myMove = {}
-- Stay (Attack)
function myMove.stay(button)
    if button == nil then
        emu.frameadvance()
    else
        myInput({button})
    end
end
-- Forward (Attack)
function myMove.forward(direction, button)
    if button == nil then
        if direction < 0 then
            myInput({"Right"})
        else
            myInput({"Left"})
        end
    else
        if direction < 0 then
            myInput({"Right", button})
        else
            myInput({"Left", button})
        end
    end
end
-- Backward (Attack)
function myMove.backward(direction, button)
    if button == nil then
        if direction > 0 then
            myInput({"Right"})
        else
            myInput({"Left"})
        end
    else
        if direction > 0 then
            myInput({"Right", button})
        else
            myInput({"Left", button})
        end
    end
end
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
-- Forward Jump (Attack)
function myMove.forwardjump(direction, button)
    if button == nil then
        if direction < 0 then
            myInput({"Right", "Up"})
        else
            myInput({"Left", "Up"})
        end
    else
        if direction < 0 then
            myInput({"Right", "Up", button})
        else
            myInput({"Left", "Up", button})
        end
    end
end
-- Backward Jump (Attack)
function myMove.backwardjump(direction, button)
    if button == nil then
        if direction > 0 then
            myInput({"Right", "Up"})
        else
            myInput({"Left", "Up"})
        end
    else
        if direction > 0 then
            myInput({"Right", "Up", button})
        else
            myInput({"Left", "Up", button})
        end
    end
end
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
-- TigerKnee 12369K
function myMove.tigerknee(direction, button)
    if direction < 0 then
        myInput({"Down", "Left"})
        myInput({"Down"})
        myInput({"Down", "Right"})
        myInput({"Right"})
        myInput({"Up", "Right"})
        myInput({button})
    else
        myInput({"Down", "Right"})
        myInput({"Down"})
        myInput({"Down", "Left"})
        myInput({"Left"})
        myInput({"Up", "Left"})
        myInput({button})
    end
end
-- Block
function myMove.block(direction,p2Jump)
    if direction < 0 then
        if p2Jump > 0 then
            myInput({"Left"})
        else
            myInput({"Down", "Left"})
        end
    else
        if p2Jump > 0 then
            myInput({"Right"})
        else
            myInput({"Down", "Right"})
        end
    end
end
-- Take Action
-- Action List
-- 1-7: stay (+XYZABC)
-- 8-14: crouch (+XYZABC)
-- 15-21: jump (+XYZABC)
-- 22-23: forward (+Z)
-- 24-25: backward (+Z)
-- 26-32: forward jump (+XYZABC)
-- 33-39: backward jump (+XYZABC)
-- 40-45: tiger shot (XYZABC)
-- 46-48: tiger uppercut (XYZ)
-- 49-51: tiger knee (ABC)
-- 52: block
function takeAction(action)
    now = getMemory.new()
    if action == 1 then
        myMove.stay("X")
    elseif action == 2 then
        myMove.stay("Z")
    elseif action == 3 then
        myMove.stay("A")
    elseif action == 4 then
        myMove.stay("C")
    elseif action == 5 then
        myMove.crouch()
    elseif action == 6 then
        myMove.crouch("X")
    elseif action == 7 then
        myMove.crouch("Z")
    elseif action == 8 then
        myMove.crouch("A")
    elseif action == 9 then
        myMove.crouch("C")
    elseif action == 10 then
        myMove.jump()
    elseif action == 11 then
        myMove.forward(now.p1X - now.p2X)
    elseif action == 12 then
        myMove.forward(now.p1X - now.p2X, "Z")
    elseif action == 13 then
        myMove.backward(now.p1X - now.p2X)
    elseif action == 14 then
        myMove.backward(now.p1X - now.p2X, "Z")
    elseif action == 15 then
        myMove.forwardjump(now.p1X - now.p2X)
    elseif action == 16 then
        myMove.backwardjump(now.p1X - now.p2X)
    elseif action == 17 then
        myMove.tiger(now.p1X - now.p2X, "Y")
    elseif action == 18 then
        myMove.tiger(now.p1X - now.p2X, "B")
    elseif action == 19 then
        myMove.uppercut(now.p1X - now.p2X, "Y")
    elseif action == 20 then
        myMove.tigerknee(now.p1X - now.p2X, "B")
    else
        myMove.block(now.p1X - now.p2X, now.p2Jump) 
    end
end
-- Get RAM observation and return a string with length 29
function get_ram(p2Character)
    now = getMemory.new()
    p1X = now.p1X
    p2X = now.p2X
    yDiff = now.yDiff

    if now.p1Moving > 0 then
        p1Moving = 1
    else
        p1Moving = 0
    end

    if now.p2Moving > 0 then
        p2Moving = 1
    else
        p2Moving = 0
    end

    if now.p1Jump == 1 then
        p1Rise = 1
        p1Fall = 0
    elseif now.p1Jump > 1 then
        p1Rise = 0
        p1Fall = 1
    else
        p1Rise = 0
        p1Fall = 0
    end

    if now.p2Jump == 1 then
        p2Rise = 1
        p2Fall = 0
    elseif now.p2Jump > 1 then
        p2Rise = 0
        p2Fall = 1
    else
        p2Rise = 0
        p2Fall = 0
    end

    if now.p1InAct % 2 == 0 then
        p1InAct = 0
    else
        p1InAct = 1
    end
    
    if now.p2InAct % 2 == 0 then
        p2InAct = 0
    else
        p2InAct = 1
    end

    if now.p1Crouch > 0 then
        p1Crouch = 1
    else
        p1Crouch = 0
    end
    
    if now.p2Crouch > 0 then
        p2Crouch = 1
    else
        p2Crouch = 0
    end
    ram = p1X.." "..p2X.." "..yDiff.." "..p1Moving.." "..p2Moving.." "..p1Rise.." "..p2Rise.." "..p1Fall.." "..p2Fall.." "..p1InAct.." "..p2InAct.." "..p1Crouch.." "..p2Crouch
    for key,value in pairs(p2Character) do
        ram = ram.." "..value
    end
    return ram
end
-- Get Fireball position
function get_Fireball()
    now = getMemory.new()
    return now.p2Fireball
end
-- Get HP difference between P1 and P2 as reward
function get_HP()
    now = getMemory.new()
    HP = now.p1HP - now.p2HP
    return HP
end
-- Figure out whether the game is over
function get_Done()
    now = getMemory.new()
    if now.p1Character > 254 then
        done = 1
    elseif now.p2Character > 254 then
        done = -1
    else 
        done = 0
    end
    return done
end

-- Main Loop
console.clear()
local nm_episode = 0
local characterName = {"Ryu", "Honda", "Blanka", "Guile", "Ken", "Chunli", "Zangief", "Dhalsim", "Vega", "Sagat", "M.Bison", "Barlog", "Cammy", "Hawk", "Feilong", "DeeJay"}
local playing = true
local inBattle = false
while playing do
    -- Reset game
    if inBattle == false then
        savestate.load('train.State')
        inBattle = true
        -- Set P1 to Sagat
        memory.write_u16_be(p1CharacterAddr, 9, "68K RAM")
        -- Set P2 to random fighter
        memory.write_u16_be(p2CharacterAddr, math.random(16) - 1, "68K RAM")
        emu.frameadvance()
        now = getMemory.new()
        rivalName = characterName[now.p2Character+1]
        p2Character = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}
        p2Character[now.p2Character+1] = 1
        nm_episode = nm_episode + 1
    end

    -- Loop when playing
    while inBattle do 
        if inBattle == false then
            break
        end
        now = getMemory.new()
        -- Wait till fight starts
        while now.p1X + now.p2X == 0 or now.timeRemain > 39207 do
            emu.frameadvance()
            now = getMemory.new()
        end

        now = getMemory.new()
        if now.p1InAct % 2 == 0 then
            lastHP = get_HP()
            lastFireball = get_Fireball()
            action, status, partial = tcp:receive('*l')
            action = tonumber(action)
            takeAction(action)
            client.screenshottoclipboard()
            tcp:send("0")
            feedback, status, partial = tcp:receive('*l')
            ram = get_ram(p2Character)
            HP = get_HP()
            Fireball = get_Fireball()
            reward = HP - lastHP
            if Fireball == lastFireball then
                Fireball = 0
            else
                now = getMemory.new()
                Fireball = math.abs(Fireball - now.p1X)
            end
            ram = Fireball.." "..ram
            if action == 17 or action == 18 then
                reward = reward + 30
            end 
            if action == 22 then
                reward = reward + 20
            end
            win = get_Done() -- P1 win 1, P2 win -1, else 0
            if win == 1 then
                done = 1
                inBattle = false
                console.log("Win vs "..rivalName)
                reward = 10
                tcp:send(ram.." "..reward.." "..done)
                console.log(ram.."!!!"..reward.."!!!"..done)
            elseif win == -1 then
                done = 2
                inBattle = false
                console.log("Lose vs "..rivalName)
                reward = -10
                tcp:send(ram.." "..reward.." "..done)
                console.log(ram.."!!!"..reward.."!!!"..done)
            else
                done = 0
                tcp:send(ram.." "..reward.." "..done)
            end
            emu.frameadvance()
            if nm_episode % 5 == 0 then
                console.clear()
            end
        else
            emu.frameadvance()
        end
    end
    emu.frameadvance()
end

tcp:close()
