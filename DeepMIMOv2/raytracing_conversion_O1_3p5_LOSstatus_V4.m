clearvars
% close all
clc

%%% Brief description:
%---------------------
% This code converts the output data of the Wireless InSite stored as
% '.p2m' files into MATLAB data stored as '.mat' files

%%% Instructions:
%-----------------
% 1) Move this '.m' file to the main directory
% 2) Create a new folder named "RayTracing Scnearios" in the main directory
% 3) Create two subfolders inside the "RayTracing Scnearios" folder, one is
% named with the name of the scenario you will read and the other is named
% with the scenario name you will write
% 4) move all the Wireless InSite '.p2m' files of the scenario you will
% read inside its corresponding subfolder
% N.B.: main '.p2m' files needed are 'dod', 'doa', 'cir', and 'pl' files
% 5) Change only the values of the input parameters in this '.m' file
% 6) Run this '.m' file!
% 7) Fetch the the output '.mat' files of the scenario you wrote from its
% corresponding subfolder

fprintf('This message is posted at date and time %s\n', datestr(now,'dd-mmm-yyyy HH:MM:SS.FFF'))
%% Input parameters

%--- Wireless InSite scenario name mapping parameters ---%
scenario_read = 'DeepSenseScenario1_v3'; %Wireless InSite input scenario name
scenario_write = 'DeepSenseScenario1_v3'; %Output scenario name

carrier_freq = (60)*1e9; %Hz
transmit_power = 0; %dBm

%--- Scenario parameters ---%
%%% Transmitter info
TX_ID_BS = [1]; % Transmitter ID number matching order (Base Stations for O1 scenarios)
TX_ID= TX_ID_BS;
num_TX_BS =numel(TX_ID_BS); % Number of BS %%%%%%%Exported directly to DeepMIMO
num_BS=numel(TX_ID_BS); % Number of BS %%%%%%%Exported directly to DeepMIMO

%%% Receiver info
RX_grids_ID_user = [2, 3]; % user ID number matching order
RX_grids_ID_BS = [1]; % BS receiver ID number matching order
RX_grids_ID = [RX_grids_ID_user,RX_grids_ID_BS]; % Receiver ID number matching order
%BS_RX_row_ID = [(5204:1:(5204+numel(RX_grids_ID_BS)-1));(5204:1:(5204+numel(RX_grids_ID_BS)-1));ones(1,numel(RX_grids_ID_BS))];
BS_RX_row_ID = [(1:1:(1+numel(RX_grids_ID_BS)-1));(1:1:(1+numel(RX_grids_ID_BS)-1));ones(1,numel(RX_grids_ID_BS))];
BS_grids=BS_RX_row_ID.'; % BS Grid Info %%%%%%%Exported directly to DeepMIMO
user_grids=[1,411,41;412,881,51]; % User Grid Info %%%%%%%Exported directly to DeepMIMO
RX_grids=[user_grids;BS_grids]; % ALL RX Grid Info

% Each row is the information associated with each of the 3 user grids:
% from user row R#"first element" to user row R#"second element" with "third element" users per row. 

%% Read and Write the Direction of Departures (DoDs) and Complex Impulse Responses (CIRs) for every BS

TX_ID_str = number2stringTX(TX_ID);
TX_ID_BS_str = number2stringTX(TX_ID_BS);
TX_ID_BS2_str = number2stringRX(TX_ID_BS);

% num_TX=numel(TX_ID); % Number of transmitters
RX_grids_ID_str = number2stringRX(RX_grids_ID);
RX_grids_ID_user_str = number2stringRX(RX_grids_ID_user);
RX_grids_ID_BS_str = number2stringRX(RX_grids_ID_BS);

num_RX_grids=numel(RX_grids_ID); % Number of RX grids
num_user_grids=numel(RX_grids_ID_user); % Number of user RX grids
num_BS_grids = numel(RX_grids_ID_BS); % Number of BS RX grids

for ii=1 %1:1:num_BS % For each BS
    disp('Finding number of user grid receiver points for BS-user channels ...')
    num_points=0;
    for jj=1:1:num_user_grids % For each user grid
        
        % Read DoD files
        filename_DoD=strcat('Raytracing_scenarios/',scenario_read,'/',scenario_read,'.dod.t001_',TX_ID_BS_str{ii},'.r',RX_grids_ID_user_str{jj},'.p2m');
        DoD_data=importdata(filename_DoD);
        num_points=num_points+DoD_data.data(1);
    end
end

for ii=1 %1:1:num_TX_BS % For each BS transmitter
    disp('Finding number of BS grid receiver points for BS-BS channels ...')
    BB_num_points=0;
    for jj=1:1:num_BS_grids % For each BS receiver
        
        % Read DoD files
        filename_DoD=strcat('Raytracing_scenarios/',scenario_read,'/',scenario_read,'.dod.t001_',TX_ID_BS_str{ii},'.r',RX_grids_ID_BS_str{jj},'.p2m');
        DoD_data=importdata(filename_DoD);
        BB_num_points=BB_num_points+DoD_data.data(1);
    end
end

%% Read and write the LoS tags from every BS to every user

for ii= 1:1:num_BS % For each BS
    disp(['Reading and writing the LoS tag files for BS# ' num2str(ii) ' out of ' num2str(num_BS) ' BSs (BS-user) ...'])
    LOS_tag_array_full=[];
    fprintf('This message is posted at time %s\n', datestr(now,'HH:MM:SS.FFF'))
    for jj=1:1:num_user_grids % For each user grid
        disp('  ');
        disp(['User grid ' num2str(jj)  ' :       ']);
        % Read PATHS files
        filename_PATHS=strcat('Raytracing_scenarios/',scenario_read,'/',scenario_read,'.paths.t001_',TX_ID_BS_str{ii},'.r',RX_grids_ID_user_str{jj},'.p2m');
        PATHS_data=importdata(filename_PATHS,' ', 1e8);
        No_lines = length(PATHS_data); %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        Starting_line_idx = 23; %Skip 22 lines of headers %first line of [No. of RX_point path_num]
        count = 0;
        %size of LOS_tag_array is the number of users/BS per user/BS grid
        No_users_per_currentgrid = (user_grids(jj,2)-user_grids(jj,1)+1)*user_grids(jj,3);
        LOS_tag_array = -2*ones(1,No_users_per_currentgrid); %Status: "LoSstatus is not checked yet".  %%%%%%%%%%%%%%%%%%
        count_update=0;
        reverseStr=0;
        for cc=Starting_line_idx:1:No_lines %increase line index for each loop
            inter_type_1 = [];
            inter_type_2 = [];
            temp1 = str2num(PATHS_data{cc});
            if(length(temp1) == 2) %find the line with [No. of RX_point path_num]
                count = count + 1;
                if(temp1(2) == 0) % Check if the number of paths is 0 == receiver in completely blocked
                    LOS_tag_array(count) = -1;  %Status: "complete blockage". Nothing is received. %%%%%%%%%%%%%%%%%%
                else
                    path1 = str2num(PATHS_data{cc + 2});% Read first path
                    if  path1(2) == 0 % Number of reflections is zero
                        LOS_tag_array(count) = 1; %Status: "LoS exists". LoS and NLoS path(/s) are received. %%%%%%%%%%%%%%%%%%
                    else
                        %%%% New part
                        if cc + 7 + path1(2) > No_lines %Last path recorded in the file
                            path1 = PATHS_data{cc + 3};% It has to be after path2
                            string1 = split( path1, '-' );
                            inter_char = {'R','D','T','DS'};
                            for qq = 1:4 % Path must have R, T, or D to be NLOS
                                inter_type_1(:,qq) = cellfun(@(y) strcmp(y, inter_char{qq}), string1);
                            end
                            if any(inter_type_1(:))
                                LOS_tag_array(count) = 0; %Status: "LoS blockage". Only NLoS path(/s) is(/are) received. %%%%%%%%%%%%%%%%%%
                            else
                                LOS_tag_array(count) = 1; %Status: "LoS exists". LoS and NLoS path(/s) are received. %%%%%%%%%%%%%%%%%%
                            end
                        else %Not yet the last path recorded in the file
                            path2 = PATHS_data{ cc + 7 + path1(2) };
                            path1 = PATHS_data{cc + 3};% It has to be after path2
                            string1 = split( path1, '-' );
                            string2 = split( path2, '-' );
                            inter_char = {'R','D','T','DS'};
                            for qq = 1:4 % Path must have R, T, or D to be NLOS
                                inter_type_1(:,qq) = cellfun(@(y) strcmp(y, inter_char{qq}), string1);
                            end
                            for qq = 1:4 % This is for the WI bug when the 2nd path is LOS
                                inter_type_2(:,qq) = cellfun(@(y) strcmp(y, inter_char{qq}), string2);
                            end
                            if any(inter_type_1(:)) && any(inter_type_2(:))
                                LOS_tag_array(count) = 0; %Status: "LoS blockage". Only NLoS path(/s) is(/are) received. %%%%%%%%%%%%%%%%%%
                            else
                                LOS_tag_array(count) = 1; %Status: "LoS exists". LoS and NLoS path(/s) are received. %%%%%%%%%%%%%%%%%%
                            end
                        end
                    end
                end
            end
            count_update = count_update + 1;
            perc_update = 100 * count_update /(No_lines-Starting_line_idx);
            msg = sprintf('- Percent done: %3.1f', perc_update); %Don't forget this semicolon
            fprintf([reverseStr, msg]);
            reverseStr = repmat(sprintf('\b'), 1, length(msg));
        end
        
        % Concatenate LoS tag arrays
        LOS_tag_array_full=[LOS_tag_array_full LOS_tag_array];
    end
    disp('          ');
    fprintf('This message is posted at time %s\n', datestr(now,'HH:MM:SS.FFF'))
    disp('          ');
    
    % Name of the output LoS files
    sfile_LoS_tag=strcat('Raytracing_scenarios/',scenario_write,'/',scenario_write,'.',num2str(ii),'.LoS.mat');
    % Concatenate the number of points to the final output array
    LOS_tag_array_full=[num_points LOS_tag_array_full];
    % Write the LoS output files
    save(sfile_LoS_tag,'LOS_tag_array_full');
end

%% Read and write the LoS tags from every BS to every BS

for ii= 1:1:num_TX_BS % For each BS
    disp(['Reading and writing the LoS tag files for BS# ' num2str(ii) ' out of ' num2str(num_TX_BS) ' BSs (BS-BS) ...'])
    LOS_tag_array_full=[];
    fprintf('This message is posted at time %s\n', datestr(now,'HH:MM:SS.FFF'))
    for jj=1:1:num_BS_grids % For each user grid
        disp('  ');
        disp(['BS grid ' num2str(jj)  ' :       ']);
        % Read PATHS files
        filename_PATHS=strcat('Raytracing_scenarios/',scenario_read,'/',scenario_read,'.paths.t001_',TX_ID_BS_str{ii},'.r',RX_grids_ID_BS_str{jj},'.p2m');
        PATHS_data=importdata(filename_PATHS,' ', 1e8);
        No_lines = length(PATHS_data);
        Starting_line_idx = 23; %Skip 22 lines of headers
        count = 0;
        %size of LOS_tag_array is the number of users/BS per user/BS grid
        No_BSs_per_currentgrid = (BS_grids(jj,2)-BS_grids(jj,1)+1)*BS_grids(jj,3);
        LOS_tag_array = -2*ones(1,No_BSs_per_currentgrid); %Status: "LoSstatus is not checked yet".  %%%%%%%%%%%%%%%%%%
        count_update=0;
        reverseStr=0;
        for cc=Starting_line_idx:1:No_lines %increase line index for each loop
            inter_type_1 = [];
            inter_type_2 = [];
            temp1 = str2num(PATHS_data{cc});
            if(length(temp1) == 2) %find the line with [No. of RX_point path_num]
                count = count + 1;
                if(temp1(2) == 0) % Check if the number of paths is 0 == receiver in completely blocked
                    LOS_tag_array(count) = -1;  %Status: "complete blockage". Nothing is received. %%%%%%%%%%%%%%%%%%
                else
                    path1 = str2num(PATHS_data{cc + 2});% Read first path
                    if  path1(2) == 0 % Number of reflections is zero
                        LOS_tag_array(count) = 1; %Status: "LoS exists". LoS and NLoS path(/s) are received. %%%%%%%%%%%%%%%%%%
                    else
                        %%%% New part
                        if cc + 7 + path1(2) > No_lines %Last path recorded in the file
                            path1 = PATHS_data{cc + 3};% It has to be after path2
                            string1 = split( path1, '-' );
                            inter_char = {'R','D','T','DS'};
                            for qq = 1:4 % Path must have R, T, or D to be NLOS
                                inter_type_1(:,qq) = cellfun(@(y) strcmp(y, inter_char{qq}), string1);
                            end
                            if any(inter_type_1(:))
                                LOS_tag_array(count) = 0; %Status: "LoS blockage". Only NLoS path(/s) is(/are) received. %%%%%%%%%%%%%%%%%%
                            else
                                LOS_tag_array(count) = 1; %Status: "LoS exists". LoS and NLoS path(/s) are received. %%%%%%%%%%%%%%%%%%
                            end
                        else %Not yet the last path recorded in the file
                            path2 = PATHS_data{ cc + 7 + path1(2) };
                            path1 = PATHS_data{cc + 3};% It has to be after path2
                            string1 = split( path1, '-' );
                            string2 = split( path2, '-' );
                            inter_char = {'R','D','T','DS'};
                            for qq = 1:4 % Path must have R, T, or D to be NLOS
                                inter_type_1(:,qq) = cellfun(@(y) strcmp(y, inter_char{qq}), string1);
                            end
                            for qq = 1:4 % This is for the WI bug when the 2nd path is LOS
                                inter_type_2(:,qq) = cellfun(@(y) strcmp(y, inter_char{qq}), string2);
                            end
                            if any(inter_type_1(:)) && any(inter_type_2(:))
                                LOS_tag_array(count) = 0; %Status: "LoS blockage". Only NLoS path(/s) is(/are) received. %%%%%%%%%%%%%%%%%%
                            else
                                LOS_tag_array(count) = 1; %Status: "LoS exists". LoS and NLoS path(/s) are received. %%%%%%%%%%%%%%%%%%
                            end
                        end
                    end
                end
            end
            count_update = count_update + 1;
            perc_update = 100 * count_update /(No_lines-Starting_line_idx+1);
            msg = sprintf('- Percent done: %3.1f', perc_update); %Don't forget this semicolon
            fprintf([reverseStr, msg]);
            reverseStr = repmat(sprintf('\b'), 1, length(msg));
        end
        
        % Concatenate LoS tag arrays
        LOS_tag_array_full=[LOS_tag_array_full LOS_tag_array];
    end
    disp('          ');
    fprintf('This message is posted at time %s\n', datestr(now,'HH:MM:SS.FFF'))
    disp('          ');
    
    % Name of the output LoS files
    sfile_LoS_tag=strcat('Raytracing_scenarios/',scenario_write,'/',scenario_write,'.',num2str(ii),'.LoS.BSBS.mat');
    % Concatenate the number of points to the final output array
    LOS_tag_array_full=[BB_num_points LOS_tag_array_full];
    % Write the LoS output files
    save(sfile_LoS_tag,'LOS_tag_array_full');
end
disp(' ')
disp('done!')
disp(' ')
fprintf('This message is posted at date and time %s\n', datestr(now,'dd-mmm-yyyy HH:MM:SS.FFF'))

%% Local functions
function [stringarrayTX] = number2stringTX(numberarrayTX)
%number2stringTX converts the BS ID number to a string with prefix of appended zeros
stringarrayTX = cell(numel(numberarrayTX),1);
for tt=1:1:numel(numberarrayTX)
    if numberarrayTX(tt)<10
        stringarrayTX{tt} = strcat('0',num2str(numberarrayTX(tt)));
    else
        stringarrayTX{tt} = num2str(numberarrayTX(tt));
    end
end
end

function [stringarrayRX] = number2stringRX(numberarrayRX)
%number2stringRX converts the user grid ID number to a string with prefix of appended zeros
stringarrayRX = cell(numel(numberarrayRX),1);
for rr=1:1:numel(numberarrayRX)
    if numberarrayRX(rr)<10
        stringarrayRX{rr} = strcat('00',num2str(numberarrayRX(rr)));
    else
        stringarrayRX{rr} = strcat('0',num2str(numberarrayRX(rr)));
    end
end
end
