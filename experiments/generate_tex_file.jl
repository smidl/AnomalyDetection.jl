include("generate_tex_tables.jl")
include("generate_tex_graphs.jl")

intex = "/home/vit/Dropbox/Cisco/kdd2018/text/kdd2018_notables.tex"
outtex = "/home/vit/Dropbox/Cisco/kdd2018/text/kdd2018.tex"

# slurp the infile
open(intex, "r") do f
    text = readstring(f)
    # now replace the table holders with the appropriate tex strings
	# summary tables
	text = replace(text, "%__TABLE_2__", sra)
	text = replace(text, "%__TABLE_3__", smt)
	text = replace(text, "%__TABLE_4__", sraa)

	# comprehensive tables for normal AUC
	text = replace(text, "%__TABLE_5__", tsas)
	text = replace(text, "%__TABLE_6__", tas)
	text = replace(text, "%__TABLE_7__", t1as)
	text = replace(text, "%__TABLE_8__", t5as)

	# augmented AUC
	text = replace(text, "%__TABLE_9__", tsaas)
	text = replace(text, "%__TABLE_10__", taas)
	text = replace(text, "%__TABLE_11__", t1aas)
	text = replace(text, "%__TABLE_12__", t5aas)

	# times
	text = replace(text, "%__TABLE_13__", fts)
	text = replace(text, "%__TABLE_14__", pts)

	# tikz critical diagrams
	text = replace(text, "%__FIGURE_1__", tacds)
	text = replace(text, "%__FIGURE_2__", tracds)
	text = replace(text, "%__FIGURE_3__", t1acds)
	text = replace(text, "%__FIGURE_4__", t5acds)

	# finally, write into the outfile
	open(outtex, "w") do f
	    write(f, text)
	end

end
