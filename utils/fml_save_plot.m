function fml_save_plot(plot_fn)
% FML_SAVE_PLOT(plot_fn)
% function to quickly save current figure to file
%   using export_fig

  set(gcf, 'Position', [100 100 800 800]);
  axis square
  set(gca,'LooseInset',get(gca,'TightInset'))
  set(gcf,'Color','w')
  set(findall(gcf,'type','text'),'FontSize',14)
  export_fig(plot_fn);

end
