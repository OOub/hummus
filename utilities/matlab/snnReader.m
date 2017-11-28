function [output] = snnReader(filename)
    fileID = fopen(filename,'rb');

    timestamp = []; delay = []; potential = []; threshold = []; preN = []; postN = [];

    while ~feof(fileID)
        currentPosition = ftell(fileID);
        exitTest = fread(fileID,1);
        if ~isempty(exitTest)
            fseek(fileID,currentPosition,'bof');
            timestamp(end+1) = fread(fileID,1,'float32');
            delay(end+1) = fread(fileID,1,'float32');
            potential(end+1) = fread(fileID,1,'float32');
            threshold(end+1) = fread(fileID,1,'float32');
            preN(end+1) = fread(fileID,1,'int16');
            postN(end+1) = fread(fileID,1,'int16');
        else
            disp('finished reading')
            break;
        end
    end
    fclose(fileID);
    variableNames = {'timestamp','delay','preN','postN','potential','threshold'};
    output = table(timestamp',delay',preN',postN',potential',threshold','VariableNames',variableNames);
end