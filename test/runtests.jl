using Judananas
using Test
using RecipesBase
using HDF5

@testset "Judananas" begin

@testset "IntervalSet construction" begin
    ep = IntervalSet([10, 0], [15, 5])
    @test starts(ep) == [0.0, 10.0] && stops(ep) == [5.0, 15.0]
    @test length(ep) == 2
    @test tot_length(ep) == 10.0
    @test collect(ep) == [(0.0, 5.0), (10.0, 15.0)]

    # overlapping epochs are merged, touching ones kept apart
    @test IntervalSet([0, 3], [5, 8]) == IntervalSet(0, 8)
    @test length(IntervalSet([0, 1], [1, 2])) == 2

    @test_throws ArgumentError IntervalSet([0], [-1])
    @test_throws ArgumentError IntervalSet([0, 3], [5, 8]; label = ["a", "b"])
    @test_throws DimensionMismatch IntervalSet([0, 1], [1, 2]; label = ["a"])

    # metadata follows the sort
    ep = IntervalSet([20, 0, 10], [25, 5, 15]; cond = ["C", "A", "B"])
    @test ep[:cond] == ["A", "B", "C"]
    @test ep[2] == IntervalSet(10, 15; cond = ["B"])
    @test ep[[3, 1]][:cond] == ["A", "C"]          # stays sorted
    ep[:block] = [1, 1, 2]
    @test ep[:block] == [1, 1, 2]

    @test 3 in ep && !(7 in ep) && 15 in ep
    @test isempty(IntervalSet())
    @test IntervalSet([0 1; 2 3]) == IntervalSet([0, 2], [1, 3])
end

@testset "IntervalSet algebra" begin
    a = IntervalSet([0, 10], [5, 20])
    b = IntervalSet([3, 12, 30], [11, 15, 40])

    @test intersect(a, b) == IntervalSet([3, 10, 12], [5, 11, 15])
    @test union(a, b) == IntervalSet([0, 30], [20, 40])
    @test setdiff(a, b) == IntervalSet([0, 11, 15], [3, 12, 20])
    @test setdiff(b, a) == IntervalSet([5, 30], [10, 40])

    # touching edges do not intersect, but do unite
    @test isempty(intersect(IntervalSet(0, 1), IntervalSet(1, 2)))
    @test union(IntervalSet(0, 1), IntervalSet(1, 2)) == IntervalSet(0, 2)

    # identities
    @test intersect(a, a) == a
    @test isempty(setdiff(a, a))
    @test union(intersect(a, b), setdiff(a, b)) == a
    @test isempty(intersect(a, IntervalSet()))

    ep = IntervalSet([0, 5.5, 10], [5, 7, 10.2])
    @test merge_close_intervals(ep, 1) == IntervalSet([0, 10], [7, 10.2])
    @test drop_short_intervals(ep, 1) == IntervalSet([0, 5.5], [5, 7])
    @test drop_long_intervals(ep, 1) == IntervalSet(10, 10.2)
end

@testset "Ts and Tsd" begin
    x = Ts([3.0, 1.0, 2.0])
    @test times(x) == [1.0, 2.0, 3.0]
    @test time_support(x) == IntervalSet(1, 3)
    @test x[2] == 2.0
    @test length(x[1:2]) == 2

    x = Ts(0:0.5:10; time_support = IntervalSet([0, 6], [2, 7]))
    @test times(x) == [0, 0.5, 1, 1.5, 2, 6, 6.5, 7]

    # touching epochs never duplicate a sample
    x = Ts([0.5, 1.0, 1.5]; time_support = IntervalSet([0, 1], [1, 2]))
    @test times(x) == [0.5, 1.0, 1.5]

    d = Tsd([2.0, 1.0], [20, 10])
    @test times(d) == [1.0, 2.0] && values(d) == [10, 20]
    @test d[2] == 20

    m = Tsd(1:4, reshape(1:8, 4, 2); columns = ["x", "y"])
    @test size(m) == (4, 2)
    @test values(m[:, "y"]) == 5:8
    @test values(m[:, 1]) == 1:4
    @test columns(m[:, [2]]) == ["y"]
    @test m[2] == [2, 6]
    @test values(map(v -> 2v, m)) == 2 .* reshape(1:8, 4, 2)
    @test_throws DimensionMismatch Tsd(1:3, rand(4))
    @test_throws KeyError m[:, "z"]

    tens = Tsd(1:5, rand(5, 3, 2))
    @test size(tens[2:3]) == (2, 3, 2)
end

@testset "restrict" begin
    x = Ts(0:1:20)
    ep = IntervalSet([2, 10], [4, 12])
    r = restrict(x, ep)
    @test times(r) == [2, 3, 4, 10, 11, 12]
    @test time_support(r) == ep
    @test times(restrict(x, 5, 6)) == [5, 6]

    # support is the intersection, not ep itself
    r2 = restrict(Ts(5:8), IntervalSet(0, 6))
    @test time_support(r2) == IntervalSet(5, 6)

    d = Tsd(0:0.1:1, collect(0:10))
    @test values(restrict(d, 0.25, 0.55)) == [3, 4, 5]
end

@testset "TsGroup" begin
    g = TsGroup([[1.0, 2, 3], [0.5, 9.5], Float64[]];
                ids = [12, 7, 40], location = ["PFC", "STR", "PFC"])
    @test length(g) == 3
    @test ids(g) == [12, 7, 40]
    @test time_support(g) == IntervalSet(0.5, 9.5)
    @test rates(g) ≈ [3, 2, 0] ./ 9
    @test times(g[2]) == [0.5, 9.5]

    pfc = filter(u -> u.location == "PFC", g)
    @test ids(pfc) == [12, 40]
    @test ids(filter(u -> u.rate > 0.25 && u.id != 12, g)) == Int[]
    @test ids(g[[true, false, true]]) == [12, 40]

    g[:depth] = [100, 200, 300]
    @test g[:depth] == [100, 200, 300]
    @test_throws ArgumentError (g[:rate] = [1, 2, 3])
    @test_throws ArgumentError TsGroup([[1.0]]; ids = [1], rate = [3.0])
    @test_throws ArgumentError TsGroup([[1.0], [2.0]]; ids = [1, 1])

    r = restrict(g, 0, 2)
    @test rates(r) ≈ [2, 1, 0] ./ 1.5
    @test time_support(r) == IntervalSet(0.5, 2)

    gd = TsGroup(Dict(5 => [1.0, 2.0], 2 => [3.0]))
    @test ids(gd) == [2, 5]

    # support from Ts objects is the union of their supports
    gt = TsGroup([Ts([0, 1]), Ts([5, 6])])
    @test time_support(gt) == IntervalSet([0, 5], [1, 6])

    flat = to_tsd(g)
    @test times(flat) == [0.5, 1, 2, 3, 9.5]
    @test values(flat) == [7, 12, 12, 12, 7]
end

@testset "count" begin
    x = Ts([0.05, 0.15, 0.16, 0.95, 1.0]; time_support = IntervalSet(0, 1))
    c = count(x, 0.1)
    @test length(c) == 10
    @test times(c) ≈ 0.05:0.1:0.95
    @test values(c) == [1, 2, 0, 0, 0, 0, 0, 0, 0, 2]   # right edge in last bin
    @test sum(values(c)) == length(x)

    # incomplete last bin is dropped; bins restart at each epoch
    ep = IntervalSet([0, 2], [0.25, 2.2])
    c = count(Ts([0.01, 0.11, 0.24, 2.05, 2.15]), 0.1; ep = ep)
    @test times(c) ≈ [0.05, 0.15, 2.05, 2.15]
    @test values(c) == [1, 1, 1, 1]

    # count per epoch
    @test values(count(x; ep = IntervalSet([0, 0.5], [0.2, 1]))) == [3, 2]

    g = TsGroup([[0.1, 0.2, 0.3], [0.25]]; time_support = IntervalSet(0, 0.4))
    cg = count(g, 0.2)
    @test size(cg) == (2, 2)
    @test values(cg) == [1 0; 2 1]
    @test columns(cg) == ["1", "2"]
    @test values(count(g)) == [3 1]
end

@testset "bin_average" begin
    d = Tsd(0:0.1:0.9, collect(1.0:10.0); time_support = IntervalSet(0, 1))
    b = bin_average(d, 0.5)
    @test values(b) ≈ [3.0, 8.0]
    @test times(b) ≈ [0.25, 0.75]

    m = Tsd(0:0.1:0.9, hcat(1.0:10.0, 11.0:20.0); time_support = IntervalSet(0, 1), columns = ["a", "b"])
    bm = bin_average(m, 0.5)
    @test values(bm) ≈ [3 13; 8 18]
    @test columns(bm) == ["a", "b"]

    sparse = Tsd([0.1], [5.0]; time_support = IntervalSet(0, 1))
    v = values(bin_average(sparse, 0.5))
    @test v[1] == 5.0 && isnan(v[2])
end

@testset "value_from" begin
    pos = Tsd(0:1.0:10, collect(0:10) .* 10.0)
    spikes = Ts([0.4, 2.6, 9.9, 15.0]; time_support = IntervalSet(0, 20))
    v = value_from(spikes, pos)
    @test times(v) == [0.4, 2.6, 9.9]          # 15 is outside the data support
    @test values(v) == [0.0, 30.0, 100.0]
    @test time_support(v) == IntervalSet(0, 10)
end

@testset "threshold" begin
    speed = Tsd(0:1.0:9, [0, 6, 7, 2, 1, 8, 9, 9, 3, 0]; time_support = IntervalSet(0, 9))
    run = threshold(speed, 5)
    @test times(run) == [1, 2, 5, 6, 7]
    @test time_support(run) == IntervalSet([0.5, 4.5], [2.5, 7.5])
    @test times(threshold(speed, 5; method = :below)) == [0, 3, 4, 8, 9]
    # runs touching the edges of the support end at the support
    @test time_support(threshold(speed, -1)) == IntervalSet(0, 9)
    @test_throws ArgumentError threshold(speed, 5; method = :sideways)
end

@testset "compute_perievent" begin
    x = Ts([0.9, 1.1, 1.4, 5.0, 5.2])
    pe = compute_perievent(x, [1.0, 5.0], (-0.2, 0.3))
    @test length(pe) == 2
    @test times(pe[1]) ≈ [-0.1, 0.1]
    @test times(pe[2]) ≈ [0.0, 0.2]
    @test time_support(pe) == IntervalSet(-0.2, 0.3)
    @test pe[:event_time] == [1.0, 5.0]
    @test rates(pe) ≈ [2, 2] ./ 0.5

    g = TsGroup([[0.9, 1.1], [1.05]])
    out = compute_perievent(g, Ts([1.0]), 0.2)
    @test length(out) == 2 && times(out[2][1]) ≈ [0.05]

    # a vector of events keeps its order (e.g. trials sorted by type)
    pe = compute_perievent(x, [5.0, 1.0], (-0.2, 0.3))
    @test pe[:event_time] == [5.0, 1.0]
    @test times(pe[1]) ≈ [0.0, 0.2] && times(pe[2]) ≈ [-0.1, 0.1]

    # an IntervalSet of trials: aligned on starts, trial metadata copied
    trials = IntervalSet([5.0, 1.0], [6.0, 2.0]; trial_type = ["B", "A"], rate = [1, 2])
    pe = compute_perievent(x, trials, (-0.2, 0.3))
    @test pe[:event_time] == [1.0, 5.0] && pe[:event_stop] == [2.0, 6.0]
    @test pe[:trial_type] == ["A", "B"] && pe[:trial_rate] == [2, 1]
    @test times(pe[1]) ≈ [-0.1, 0.1]
    @test length(compute_perievent(g, trials, 0.2)) == 2
end

@testset "selection by id" begin
    g = TsGroup([[1.0], [2.0, 3.0], [4.0]]; ids = [185, 228, 7], location = ["CA1", "DG", "CA1"])
    @test times(g[id = 228]) == [2.0, 3.0]
    @test ids(g[id = [7, 185]]) == [7, 185]
    @test g[:location, id = 7] == "CA1"
    @test g[:location, id = [228, 7]] == ["DG", "CA1"]
    @test g[:location] == ["CA1", "DG", "CA1"]
    @test indexof(g, 7) == 3 && indexof(g, [228, 185]) == [2, 1]
    m = metadata(g, 228)
    @test m.id == 228 && m.location == "DG" && m.rate ≈ 2 / 3
    @test metadata(g, :location, 185) == "CA1"
    @test_throws KeyError g[id = 999]
    @test_throws KeyError indexof(g, 999)
end

@testset "metadata tables" begin
    g = TsGroup([[1.0], [2.0, 3.0], [4.0]]; ids = [185, 228, 7], location = ["CA1", "DG", "CA1"])
    t = metadata_table(g)
    @test keys(t) == (:id, :location, :rate)
    @test t.id == [185, 228, 7] && t.location == ["CA1", "DG", "CA1"]
    @test value_counts(g, :location) == ["CA1" => 2, "DG" => 1]

    ep = IntervalSet([0, 10], [5, 15]; cond = ["A", "A"])
    @test metadata_table(ep) == (start = [0.0, 10.0], stop = [5.0, 15.0], cond = ["A", "A"])
    @test value_counts(ep, :cond) == ["A" => 2]
end

@testset "rate, time_diff, ISI distribution" begin
    x = Ts([1.0, 2.0, 4.0, 7.0]; time_support = IntervalSet(0, 10))
    @test rate(x) == 0.4
    @test rate(restrict(x, 0, 2.5)) == 2 / 2.5
    @test isnan(rate(Ts(Float64[])))

    d = time_diff(x)
    @test values(d) == [1.0, 2.0, 3.0]
    @test times(d) == [1.5, 3.0, 5.5]
    @test time_support(d) == time_support(x)
    @test times(time_diff(x; align = :start)) == [1.0, 2.0, 4.0]
    @test times(time_diff(x; align = :end)) == [2.0, 4.0, 7.0]
    @test_throws ArgumentError time_diff(x; align = :middle)
    # no interval across the gap between two epochs
    y = Ts([1.0, 2.0, 11.0, 13.0]; time_support = IntervalSet([0, 10], [5, 15]))
    @test values(time_diff(y)) == [1.0, 2.0]

    h = compute_isi_distribution(x; bins = [0, 1.5, 2.5, 3])
    @test h.counts == [1, 1, 1]                      # last bin includes its right edge
    @test h.centers == [0.75, 2.0, 2.75]
    h = compute_isi_distribution(x; bins = 2)
    @test h.edges == [1.0, 2.0, 3.0] && h.counts == [1, 2]
    h = compute_isi_distribution(x; bins = 2, log_scale = true)
    @test h.edges ≈ [1.0, sqrt(3), 3.0] && h.counts == [1, 2]
    @test h.centers ≈ sqrt.(h.edges[1:end-1] .* h.edges[2:end])

    g = TsGroup([[0.0, 1.0, 3.0], [0.0, 0.5]])
    h = compute_isi_distribution(g; bins = [0, 1, 2, 3])
    @test size(h.counts) == (3, 2)
    @test h.counts[:, 1] == [0, 1, 1] && h.counts[:, 2] == [1, 0, 0]
    @test_throws ArgumentError compute_isi_distribution(x; bins = [1.0])
end

@testset "plot recipes" begin
    recipe(x) = RecipesBase.apply_recipe(Dict{Symbol,Any}(), x)[1]

    g = TsGroup([[1.0, 2.0], [1.5]]; ids = [3, 9])
    r = recipe(g)
    @test r.args[1] == [1.0, 2.0, 1.5] && r.args[2] == [1, 1, 2]
    @test r.plotattributes[:seriestype] == :scatter
    @test r.plotattributes[:yticks] == (1:2, ["3", "9"])

    # lines are broken between epochs
    d = Tsd([0.0, 1.0, 5.0, 6.0], [1, 2, 3, 4]; time_support = IntervalSet([0, 5], [1, 6]))
    t, v = recipe(d).args
    @test isnan(t[3]) && length(t) == 5
    @test v[[1, 2, 4, 5]] == [1, 2, 3, 4]

    m = Tsd(1:3, rand(3, 2); columns = ["a", "b"])
    @test recipe(m).plotattributes[:label] == ["a" "b"]

    r = recipe(IntervalSet([0, 5], [1, 6]))
    @test r.args[1] == [0, 1, 5, 6] && r.plotattributes[:seriestype] == :vspan

    @test recipe(Ts([1.0, 2.0])).args[1] == [1.0, 2.0]

    # rasters: "Unit" rows, or "Trial" rows after compute_perievent
    @test recipe(g).plotattributes[:yguide] == "Unit"
    pe = compute_perievent(Ts([0.9, 1.1, 2.0]), [1.0, 2.0], 0.5)
    @test recipe(pe).plotattributes[:yguide] == "Trial"

    # `color` colours the ticks, one colour or one per row
    colored(x, c) = RecipesBase.apply_recipe(Dict{Symbol,Any}(:seriescolor => c), x)[1].plotattributes
    a = colored(g, :red)
    @test a[:markercolor] == :red && a[:markerstrokecolor] == :red
    a = colored(g, [:red, :blue])
    @test a[:markercolor] == [:red, :red, :blue] && a[:markerstrokecolor] == [:red, :red, :blue]
    @test recipe(g).plotattributes[:markercolor] == :black
    @test colored(Ts([1.0]), :blue)[:markerstrokecolor] == :blue
end

@testset "load_file (NWB)" begin
    path = joinpath(mktempdir(), "mini.nwb")
    h5open(path, "w") do f
        f["identifier"] = "mini-session"
        f["session_description"] = "synthetic test file"
        nd(g, t) = (attrs(g)["neurodata_type"] = t; g)

        e = create_group(f, "general/extracellular_ephys/electrodes")
        e["id"] = Int32[0, 1, 2]
        e["location"] = ["CA1", "PL5", "ILA5"]
        e["x"] = [1.0, 2.0, 3.0]

        u = nd(create_group(f, "units"), "Units")
        u["id"] = Int32[10, 20, 30]
        u["spike_times"] = [0.5, 1.0, 1.5, 2.0, 0.2, 9.0]   # 3, 1, 2 spikes
        u["spike_times_index"] = [3, 4, 6]
        u["electrodes"] = Int32[2, 0, 1]                       # 0-based rows
        u["electrodes_index"] = [1, 2, 3]
        u["quality"] = [1.0 2.0 3.0; 0.1 0.2 0.3]              # NWB shape (3, 2)
        u["rate"] = [7.0, 8.0, 9.0]                            # clashes with computed :rate
        u["tags"] = ["a", "b", "c", "d"]                       # ragged column
        u["tags_index"] = [1, 3, 4]
        u["waveform_means"] = rand(5, 3)                       # skipped by name
        u["big"] = rand(1000, 3)                               # 8 kB per unit, 24 kB in all

        tr = nd(create_group(f, "intervals/trials"), "TimeIntervals")
        tr["id"] = Int32[0, 1]
        tr["start_time"] = [1.0, 5.0]
        tr["stop_time"] = [2.0, 6.0]
        tr["sound"] = [1.0, 0.0]

        ov = nd(create_group(f, "intervals/bouts"), "TimeIntervals")
        ov["start_time"] = [1.0, 1.5]
        ov["stop_time"] = [2.0, 3.0]
        ov["kind"] = ["x", "y"]

        b = nd(create_group(f, "processing/behavior/Pupil/pupil"), "TimeSeries")
        b["data"] = [0.1, 0.2, 0.3]
        b["timestamps"] = [10.0, 10.5, 11.0]

        for (name, scale) in (("LFP", 1), ("EMG", 100))
            es = nd(create_group(f, "processing/ecephys/$name/ElectricalSeries"), "ElectricalSeries")
            es["data"] = Int16.(scale .* [1 2 3 4; 11 12 13 14])  # NWB shape (4 time, 2 channels)
            es["starting_time"] = 2.0
            attrs(es["starting_time"])["rate"] = 1000.0
        end
    end

    data = load_file(path)
    @test data isa NWBFile
    @test keys(data)[1] == "units" && Set(keys(data)[2:3]) == Set(["trials", "bouts"])   # units, then intervals
    @test Set(keys(data)) == Set(["units", "trials", "bouts", "pupil",
                                  "LFP/ElectricalSeries", "EMG/ElectricalSeries"])
    @test occursin("mini-session", sprint(show, MIME"text/plain"(), data))

    g = data["units"]
    @test data["units"] === g                                       # cached
    @test ids(g) == [10, 20, 30]
    @test times(g[1]) == [0.5, 1.0, 1.5] && times(g[3]) == [0.2, 9.0]
    @test g[:location] == ["ILA5", "CA1", "PL5"]
    @test g[:x] == [3.0, 1.0, 2.0]
    @test g[:electrode] == [2, 0, 1]
    @test g[:quality] == [[1.0, 0.1], [2.0, 0.2], [3.0, 0.3]]
    @test g[:tags] == [["a"], ["b", "c"], ["d"]]
    @test g[:rate_nwb] == [7.0, 8.0, 9.0]
    @test rates(g) ≈ [3, 1, 2] ./ 8.8
    @test ids(filter(u -> u.location == "CA1", g)) == [20]

    tr = data["trials"]
    @test tr == IntervalSet([1.0, 5.0], [2.0, 6.0]; sound = [1.0, 0.0])
    @test_logs (:warn,) data["bouts"]
    @test data["bouts"] == IntervalSet(1.0, 3.0)

    p = data["pupil"]
    @test times(p) == [10.0, 10.5, 11.0] && values(p) == [0.1, 0.2, 0.3]

    lfp = data["LFP/ElectricalSeries"]
    @test size(lfp) == (4, 2)
    @test times(lfp) ≈ [2.0, 2.001, 2.002, 2.003]
    @test values(lfp)[:, 2] == [11, 12, 13, 14]                    # (time, channel)
    @test values(restrict(lfp, 2.0005, 2.0025)) == [2 12; 3 13]
    @test values(data["EMG/ElectricalSeries"])[1, 1] == 100

    @test !haskey(metadata(g), :waveform_means)
    @test length(g[:big][1]) == 1000                               # under the size limit

    # large columns are skipped, and so are the ones asked for
    small = @test_logs (:info, r"column `big` skipped") load_file(path; max_column_bytes = 10_000)["units"]
    @test !haskey(metadata(small), :big) && haskey(metadata(small), :tags)
    some = load_file(path; skip_columns = ["tags", "x"])["units"]
    @test !haskey(metadata(some), :tags) && !haskey(metadata(some), :x)
    @test haskey(metadata(some), :location)
    @test !haskey(metadata(load_file(path; skip_columns = ["sound"])["trials"]), :sound)

    @test_throws KeyError data["nope"]
    @test_throws ArgumentError load_file(joinpath(mktempdir(), "missing.nwb"))
    @test Judananas._escape_query("sub-1/a_b+c d.nwb") == "sub-1/a_b%2Bc%20d.nwb"
end

@testset "display" begin
    g = TsGroup([[1.0, 2.0], [1.5]]; location = ["PFC", "STR"])
    s = sprint(show, MIME"text/plain"(), g)
    @test occursin("2 units", s) && occursin("PFC", s)
    @test occursin("2 epochs", sprint(show, MIME"text/plain"(), IntervalSet([0, 2], [1, 3])))
    @test occursin("3 timestamps", sprint(show, MIME"text/plain"(), Ts([1, 2, 3])))
end

end
