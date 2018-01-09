clc
clear
close all

addpath('../src');

%% FEATURES
createTeacher = 1; % 1 to create a teacher signal
timeJitter = 1; % 0 to turn off time jitter and 1 to add time jitter
additiveNoise = 0; % percentage of background noise
neuronalNoise = 0; % percentage of neural noise
randomizePatternOccurence = 0; % 0 to turn off and 1 to randomize
numberOfNeurons = 27; 

%% PARAMETERS
numberOfPatterns = 4;
repetitions = 2000;
timeBetweenIntervals = 10;
timejitterInterval = [0,9.5]; %standard deviation for the jitter

%% PATTERN CREATION
pattern1 = [0;1;2;3];
time1 = [5;10;12;24];

pattern2 = [4;5;6;7];
time2 = [1;4;5;10];

pattern3 = [10;18;1;20];
time3 = [3;5;8;20];

pattern4 = [22;26;25;23];
time4 = [5;10;15;20];

patternsCollections = struct('patterns',horzcat(pattern1,pattern2,pattern3,pattern4),'times',horzcat(time1,time2,time3,time4));
numberOfSpikesInPattern = size(patternsCollections.patterns,1);

patternIndices = repmat([1:numberOfPatterns]',repetitions+1,1);
if (randomizePatternOccurence == 1)
    patternIndices = patternIndices(randperm(size(patternIndices, 1)), :);
end

for i = 1:length(patternIndices)
    data(:,i) = patternsCollections.patterns(:,patternIndices(i));
    time(:,i) = patternsCollections.times(:,patternIndices(i));
end

data = reshape(data,size(data,1)*size(data,2),1);
time = reshape(time,size(time,1)*size(time,2),1);

shift = 0;
for i = 1:length(time)
    if (timeJitter == 1)
        time(i) = time(i) + (timejitterInterval(1) + (timejitterInterval(2) - timejitterInterval(1)).*rand(1,1)) + shift; % time jitter
    elseif (timeJitter == 0)
        time(i) = time(i) + shift; % no time jitter
    end
    
    if mod(i, numberOfSpikesInPattern) == 0
        shift = time(i) + timeBetweenIntervals;
    end
end

%% CREATING TEACHER SIGNAL FORCING SNN TO SPIKE AFTER EVERY PATTERN (SUPERVISED LEARNING)
if (createTeacher == 1)
    teacherSignal = [];
    responseNeurons = [28,30,32,34];
    cnt = 1;
    for i = 4:4:size(time)
        teacherSignal(end+1,1) = time(i)+1;
        teacherSignal(end,2) = responseNeurons(cnt);
        cnt = cnt+1;
        if (cnt > 4)
            cnt = 1;
        end
    end
    if (timeJitter == 1)
        dlmwrite(strcat(num2str(timejitterInterval(2)),'teacherSignal.txt'),teacherSignal, 'delimiter', ' ','precision','%f');
    else
        dlmwrite(strcat('clean_','teacherSignal.txt'),teacherSignal, 'delimiter', ' ','precision','%f');
    end
end

clear i pattern1 pattern2 pattern3 pattern4 time1 time2 time3 time4 shift

%% ADDING NEURONAL NOISE
numberNeuronalNoise = floor(length(data)*neuronalNoise);
if (numberNeuronalNoise > 0)
    replacementIndex = randi(length(data),numberNeuronalNoise,1);
    for i = 1:length(replacementIndex)
        data(replacementIndex(i)) = randi(max(data));
    end
end

%% ADDING BACKGROUND NOISE

numberNoiseInsertion = floor(length(data)*additiveNoise/100);
if (numberNoiseInsertion > 0)
    insertionIndex = randi(length(data),numberNoiseInsertion,1);
    for i = 1:length(insertionIndex)
        neuronNoise = randi(numberOfNeurons);
        tmin = time(insertionIndex(i));
        tmax = time(insertionIndex(i)+1);
        timeNoise=tmin+rand(1,1)*(tmax-tmin);
        
        tp = data(insertionIndex(i):end,:);
        data(insertionIndex(i),:) = neuronNoise;
        data(insertionIndex(i)+1:end+1,:) = tp;

        tp1 = time(insertionIndex(i):end,:);
        time(insertionIndex(i),:) = timeNoise;
        time(insertionIndex(i)+1:end+1,:) = tp1;
    end
end

%% SAVING TO TEXT FILE
data_matrix = [time,data];
if (timeJitter == 1)
    if (randomizePatternOccurence == 1)
        dlmwrite(strcat("random_",num2str(timejitterInterval(2)),"timeJitter",num2str(additiveNoise),"bn",num2str(neuronalNoise),"nn",num2str(numberOfPatterns),'fakePatterns_snnTest_',num2str(repetitions),'reps_',num2str(timeBetweenIntervals),'msInterval.txt'), data_matrix, 'delimiter', ' ','precision','%f')
    else
        dlmwrite(strcat(num2str(timejitterInterval(2)),"timeJitter",num2str(additiveNoise),"bn",num2str(neuronalNoise),"nn",num2str(numberOfPatterns),'fakePatterns_snnTest_',num2str(repetitions),'reps_',num2str(timeBetweenIntervals),'msInterval.txt'), data_matrix, 'delimiter', ' ','precision','%f')
    end
elseif (timeJitter == 0)
    if (randomizePatternOccurence == 1)
        dlmwrite(strcat("random_",num2str(additiveNoise),"bn",num2str(neuronalNoise),"nn",num2str(numberOfPatterns),'fakePatterns_snnTest_',num2str(repetitions),'reps_',num2str(timeBetweenIntervals),'msInterval.txt'), data_matrix, 'delimiter', ' ','precision','%f')
    else
        dlmwrite(strcat(num2str(additiveNoise),"bn",num2str(neuronalNoise),"nn",num2str(numberOfPatterns),'fakePatterns_snnTest_',num2str(repetitions),'reps_',num2str(timeBetweenIntervals),'msInterval.txt'), data_matrix, 'delimiter', ' ','precision','%f')
    end
end