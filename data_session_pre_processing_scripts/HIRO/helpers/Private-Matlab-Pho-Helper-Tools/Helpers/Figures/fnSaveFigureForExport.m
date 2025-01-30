function [export_result] = fnSaveFigureForExport(fig_h, figPath, should_export_fig, should_export_eps, should_export_pdf, should_export_png)
% fnSaveFigureForExport: performs export to disk of a provided figure in one or more output formats
	% fig_h: the handle to the figure
	% figPath: the path string to the figure without an extension.

	
	% Position the figure:
% 	fig_h.Parent.OuterPosition = [0 0 4 6];
	
	% Default values for optional parameters
	if ~exist('should_export_fig','var')
		should_export_fig = true;
	end
	if ~exist('should_export_png','var')
		should_export_png = false;
	end
	if ~exist('should_export_eps','var')
		should_export_eps = false;
	end
	if ~exist('should_export_pdf','var')
		should_export_pdf = true;
	end
	enable_vector_pdf_output = false; % Explicitly enable vector PDF output if that's desired. It's very slow
	
	% Perform requested exports
	if should_export_fig
		export_result.fig = [figPath '.fig'];
		savefig(export_result.fig)
	end
	if should_export_png
		export_result.png = [figPath '.png'];
		% Requires R2020a or later
		exportgraphics(fig_h, export_result.png,'Resolution',300)
	end
	
	if should_export_eps
		export_result.eps = [figPath '.eps'];
		exportgraphics(fig_h, export_result.eps)
	end
	
	if should_export_pdf
		export_result.pdf = [figPath '.pdf'];
		% Requires R2020a or later
		if enable_vector_pdf_output
			exportgraphics(fig_h, export_result.pdf,'ContentType','vector','BackgroundColor','none');
		else
			exportgraphics(fig_h, export_result.pdf,'BackgroundColor','none');
		end
	end
end

