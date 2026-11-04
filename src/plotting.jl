# ---------------------------------------------------------------------------
# Plot recipes (RecipesBase): `using Plots; plot(x)` works on every object.
# Judananas itself only depends on the lightweight RecipesBase package.
# ---------------------------------------------------------------------------

# Insert NaN between samples belonging to different epochs, so line plots
# do not draw across gaps in the time support.
function _with_gaps(x::Tsd)
    d = Float64.(reshape(x.d, length(x.t), :))
    isempty(x.t) && return x.t, d
    ep = x.time_support
    epoch_of(t) = searchsortedlast(ep.start, t)
    rows = Int[]
    t = Float64[]
    for k in eachindex(x.t)
        if k > 1 && epoch_of(x.t[k]) != epoch_of(x.t[k-1])
            push!(rows, 0); push!(t, NaN)
        end
        push!(rows, k); push!(t, x.t[k])
    end
    out = fill(NaN, length(rows), size(d, 2))
    for (i, k) in enumerate(rows)
        k > 0 && (out[i, :] = d[k, :])
    end
    return t, size(out, 2) == 1 ? vec(out) : out
end

# Continuous data: one line per column, broken across epochs.
@recipe function f(x::Tsd)
    ndims(x.d) <= 2 || error("plotting is only defined for vector or matrix Tsd")
    xguide --> "Time (s)"
    named = ndims(x.d) == 2 && !isempty(x.columns)
    legend --> named
    label --> (named ? permutedims(x.columns) : "")
    t, d = _with_gaps(x)
    t, d
end

# Events: one tick per timestamp.
@recipe function f(x::Ts)
    seriestype --> :scatter
    markershape --> :vline
    markersize --> 10
    c = get(plotattributes, :seriescolor, :black)    # `color` colours the ticks
    markercolor --> c
    markerstrokecolor --> c
    legend --> false
    yticks --> false
    ylims --> (0.5, 1.5)
    xguide --> "Time (s)"
    x.t, ones(length(x.t))
end

# Raster: one row per unit (or per trial after compute_perievent, recognised by
# its :event_time metadata).
#
# `color` colours the ticks: one colour for the whole raster, or a vector with one
# colour per row (e.g. by trial type). `yflip = true` puts the first row on top.
@recipe function f(g::TsGroup)
    n = length(g)
    xs = reduce(vcat, (u.t for u in g.units); init = Float64[])
    ys = reduce(vcat, (fill(i, length(u)) for (i, u) in enumerate(g.units)); init = Int[])
    seriestype --> :scatter
    markershape --> :vline
    markersize --> max(2, min(8, 300 / max(n, 1)))
    c = get(plotattributes, :seriescolor, :black)
    if c isa AbstractVector && length(c) == n && n > 0
        c = reduce(vcat, (fill(c[i], length(u)) for (i, u) in enumerate(g.units)); init = eltype(c)[])
        seriescolor := :black      # per-spike colours go to the markers
    end
    # vline markers are drawn with their stroke, so both colours follow `color`
    markercolor --> c
    markerstrokecolor --> c
    legend --> false
    xguide --> "Time (s)"
    yguide --> (haskey(g.metadata, :event_time) ? "Trial" : "Unit")
    ylims --> (0.5, n + 0.5)
    if 0 < n <= 30
        yticks --> (1:n, string.(g.ids))
    end
    xs, ys
end

# Epochs: shaded vertical bands. Overlay with `plot!(ep)`.
@recipe function f(ep::IntervalSet)
    seriestype --> :vspan
    fillalpha --> 0.2
    linewidth --> 0
    label --> ""
    vec(permutedims(hcat(ep.start, ep.stop)))   # [s1, e1, s2, e2, ...]
end
