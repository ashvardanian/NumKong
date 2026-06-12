# Options affecting formatting.
with section("format"):
    # How wide to allow formatted cmake files
    line_width = 120

    # How many spaces to tab for indent
    tab_size = 4

    # If true, separate flow control names from their parentheses with a space
    separate_ctrl_name_with_space = True

    # If true, separate function names from parentheses with a space
    separate_fn_name_with_space = False

    # If a statement is wrapped to more than one line, than dangle the closing
    # parenthesis on its own line.
    dangle_parens = True

    # Allow many positional arguments before forcing vertical layout
    # (default 6 splits target_compile_definitions onto one-per-line)
    max_pargs_hwrap = 40

    # Allow more sub-groups before forcing vertical layout
    max_subgroups_hwrap = 4

    # Allow horizontal wrapping to span more lines (default 2 forces
    # one-per-line when word-wrapping exceeds 2 lines)
    max_lines_hwrap = 8

# Options affecting comment handling.
with section("markup"):
    # Do not reflow comment text
    enable_markup = False
