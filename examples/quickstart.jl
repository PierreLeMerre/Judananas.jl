# Judananas quickstart on synthetic data.
# Run from the package folder:  julia --project=examples examples/quickstart.jl
# (first time: julia --project=examples -e 'using Pkg; Pkg.develop(path="."); Pkg.add("Plots")')

using Pkg
Pkg.develop(path = "/Users/pierre/Documents/Julia/Judananas")
using Plots
using Random

Random.seed!(1)

# --- a fake session: 600 s, 20 units, 40 trials of a 0.5 s tone -----------
session = IntervalSet(0, 600)
tone_on = 20 .+ 14 .* (0:39) .+ 4 .* rand(40)    # one tone every ~14 s
tones = IntervalSet(tone_on, tone_on .+ 0.5; frequency = rand([4000, 8000, 16000], 40))

function poisson_train(rate, ep)
    t = Float64[]
    for (s, e) in ep
        n = round(Int, rate * (e - s) * 2) + 10
        x = s .+ cumsum(randexp(n) ./ rate)
        append!(t, x[x .<= e])
    end
    return t
end

spikes = map(1:20) do i
    base = poisson_train(2 + 8rand(), session)
    evoked = i <= 8 ? poisson_train(30, tones) : Float64[]   # units 1-8 respond
    sort!(vcat(base, evoked))
end

units = TsGroup(spikes; ids = 101:120, time_support = session,
                location = [i <= 10 ? "AUd" : "PFC" for i in 1:20],
                depth = rand(200:20:1200, 20))
display(units)

# --- core operations ------------------------------------------------------
aud = filter(u -> u.location == "AUd" && u.rate > 3, units)
println("\nAUd units above 3 Hz: ", ids(aud))

during = restrict(units, tones)               # spikes during tones only
println("Mean rate during tones: ", round.(rates(during)[1:8], digits = 1))

counts = count(units, 0.05)                    # 50 ms bins, bins × units
println("Binned counts: ", size(counts))

pe = compute_perievent(units[1], tone_on, (-0.5, 1.0))   # trials of unit 101

# a fake running speed and the epochs where the animal runs
speed = Tsd(0:0.1:600, abs.(cumsum(randn(6001))) ./ 20; time_support = session)
running = time_support(threshold(speed, 5))
println("Running: ", length(running), " epochs, ", round(tot_length(running), digits = 1), " s")

# --- plots ----------------------------------------------------------------
p1 = plot(restrict(units, 100, 110), title = "Raster, 100–110 s")
plot!(p1, intersect(tones, IntervalSet(100, 110)), color = :orange)

p2 = plot(pe, title = "Unit 101 aligned to tone onset", yguide = "Trial")
plot!(p2, IntervalSet(0, 0.5), color = :orange)

c = count(pe, 0.025)                           # bins × trials
psth = Tsd(times(c), vec(sum(values(c), dims = 2)) ./ (0.025 * length(pe)))
p3 = plot(psth, title = "PSTH", yguide = "Rate (Hz)", linewidth = 2)

p4 = plot(restrict(speed, 0, 120), title = "Speed and running epochs", yguide = "cm/s")
plot!(p4, intersect(running, IntervalSet(0, 120)), color = :green)

fig = plot(p1, p2, p3, p4, layout = (2, 2), size = (1000, 700))
savefig(fig, "judananas_quickstart.png")
println("Saved judananas_quickstart.png")
