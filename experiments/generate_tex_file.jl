include("generate_tex_tables.jl")

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
	text = replace(text, "%__TABLE_5__", mas)
	text = replace(text, "%__TABLE_6__", tas)
	text = replace(text, "%__TABLE_7__", t1as)
	text = replace(text, "%__TABLE_8__", t5as)

	# augmented AUC
	text = replace(text, "%__TABLE_9__", maas)
	text = replace(text, "%__TABLE_10__", taas)
	text = replace(text, "%__TABLE_11__", t1aas)
	text = replace(text, "%__TABLE_12__", t5aas)

	# times
	text = replace(text, "%__TABLE_13__", fts)
	text = replace(text, "%__TABLE_14__", pts)


	# finally, write into the outfile
	open(outtex, "w") do f
	    write(f, text)
	end

end
