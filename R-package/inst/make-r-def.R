# [description]
#     Create a definition file (.def) from a .dll file, using objdump. This
#     is used by FindLibR.cmake when building the R package with MSVC.
#
# [usage]
#
#     Rscript make-r-def.R something.dll something.def
#
# [references]
#    * https://www.cs.colorado.edu/~main/cs1300/doc/mingwfaq.html

args <- commandArgs(trailingOnly = TRUE)

IN_DLL_FILE <- args[[1L]]
OUT_DEF_FILE <- args[[2L]]
DLL_BASE_NAME <- basename(IN_DLL_FILE)

message(sprintf("Creating '%s' from '%s'", OUT_DEF_FILE, IN_DLL_FILE))

# system() will not raise an R exception if the process called
# fails. Wrapping it here to get that behavior.
#
# system() introduces a lot of overhead, at least on Windows,
# so trying processx if it is available
.pipe_shell_command_to_stdout <- function(command, args, out_file) {
    has_processx <- suppressMessages({
        suppressWarnings({
            require("processx")  # nolint
        })
    })
    if (has_processx) {
        p <- processx::process$new(
            command = command
            , args = args
            , stdout = out_file
            , windows_verbatim_args = FALSE
        )
        invisible(p$wait())
    } else {
        message(paste0(
            "Using system2() to run shell commands. Installing "
            , "'processx' with install.packages('processx') might "
            , "make this faster."
        ))
        exit_code <- system2(
            command = command
            , args = shQuote(args)
            , stdout = out_file
        )
        if (exit_code != 0L) {
            stop(paste0("Command failed with exit code: ", exit_code))
        }
    }
    return(invisible(NULL))
}

# use objdump to dump all the symbols
OBJDUMP_FILE <- "objdump-out.txt"
.pipe_shell_command_to_stdout(
    command = "objdump"
    , args = c("-p", IN_DLL_FILE)
    , out_file = OBJDUMP_FILE
)

objdump_results <- readLines(OBJDUMP_FILE)
result <- file.remove(OBJDUMP_FILE)

# Only one table in the objdump results matters for our purposes,
# see https://www.cs.colorado.edu/~main/cs1300/doc/mingwfaq.html
start_index <- which(
    grepl(
        pattern = "[Ordinal/Name Pointer] Table"
        , x = objdump_results
        , fixed = TRUE
    )
)
empty_lines <- which(objdump_results == "")
end_of_table <- empty_lines[empty_lines > start_index][1L]

# Read the contents of the table
exported_symbols <- objdump_results[(start_index + 1L):end_of_table]
exported_symbols <- gsub("\t", "", exported_symbols, fixed = TRUE)
exported_symbols <- gsub(".*\\] ", "", exported_symbols)
exported_symbols <- gsub(" ", "", exported_symbols, fixed = TRUE)

# Write R.def file
writeLines(
    text = c(
        paste0("LIBRARY \"", DLL_BASE_NAME, "\"")
        , "EXPORTS"
        , exported_symbols
    )
    , con = OUT_DEF_FILE
    , sep = "\n"
)
message(sprintf("Successfully created '%s'", OUT_DEF_FILE))
