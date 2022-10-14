list = dir('*.fig');
for i=1:size(list, 1)
    openfig(list(i).name);
    saveas(gcf,[list(i).name(1:end-4), '.eps'],'epsc');
    close all;
end