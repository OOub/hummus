% snnReader.m

% Created by Omar Oubari 
% PhD - Institut de la Vision
% Email: omar.oubari@inserm.fr

% Last Version: 18/06/2018

% Information: snnReader is a function that reads a binary file originating
% from the Nour spiking neural network simulators
% bool = false for spikeLogger file and bool = true for learningLogger file


function [output] = snnReader(filename, bool)    
    fileID = fopen(filename,'rb');
    if bool == false
        disp("reading spike logger")
        timestamp = []; delay = []; potential = []; preN = []; postN = []; layerID = []; rfID = []; weight = []; X = []; Y = [];
        
        while ~feof(fileID)
            currentPosition = ftell(fileID);
            exitTest = fread(fileID,1);
            if ~isempty(exitTest)
                fseek(fileID,currentPosition,'bof');
                timestamp(end+1) = fread(fileID,1,'float64');
                delay(end+1) = fread(fileID,1,'float32');
                weight(end+1) = fread(fileID,1,'float32');
                potential(end+1) = fread(fileID,1,'float32');
                preN(end+1) = fread(fileID,1,'int16');
                postN(end+1) = fread(fileID,1,'int16');
                layerID(end+1) = fread(fileID,1,'int16');
                rfID(end+1) = fread(fileID,1,'int16');
                X(end+1) = fread(fileID,1,'int16');
                Y(end+1) = fread(fileID,1,'int16');
            else
                disp("finished reading")
                break;
            end
        end
        fclose(fileID);
        
        variableNames = {'timestamp','delay','weight','preN','postN','potential','layerID','rfID','X','Y'};
        output = table(timestamp',delay',weight',preN',postN',potential',layerID',rfID',X',Y','VariableNames',variableNames);
    elseif bool == true
        disp("reading learning logger")
        output = {};
        plasticNeurons = [];

        while ~feof(fileID)
            currentPosition = ftell(fileID);
            exitTest = fread(fileID,1);
            if ~isempty(exitTest)
                fseek(fileID,currentPosition,'bof');
                bitSize = fread(fileID,1,'int64');
                timestamp = fread(fileID,1,'float64');
                winnerID = fread(fileID,1,'int16');
                layerID = fread(fileID,1,'int16');
                rfID = fread(fileID,1,'int16');
                
                for i = 1:(bitSize-22)/14
                    plasticNeurons(end+1,1) = fread(fileID,1,'float64');
                    plasticNeurons(end,2) = fread(fileID,1,'int16');
                    plasticNeurons(end,3) = fread(fileID,1,'int16');
                    plasticNeurons(end,4) = fread(fileID,1,'int16');
                end
                
                output{end+1,1} = struct('timestamp', timestamp, 'winnerID', winnerID, 'layerID', layerID, 'rfID', rfID, 'plasticNeurons', plasticNeurons);
                plasticNeurons = [];
            else
                disp("finished reading")
                break;
            end
        end
    end
end