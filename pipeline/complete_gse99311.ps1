$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $Root
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$Archive = "data\studies\training_candidate\GSE99311\GSE99311_RAW.tar"
$Manifest = "data\studies\training_candidate\GSE99311\manifest.json"

function Invoke-Pdac([string[]] $CliArgs) {
    & $Python -m pdac_circuit.pipeline.cli @CliArgs
    if ($LASTEXITCODE -ne 0) {
        throw "pdac command failed ($LASTEXITCODE): $($CliArgs -join ' ')"
    }
}

while (-not (Test-Path -LiteralPath $Archive) -or -not (Test-Path -LiteralPath $Manifest)) {
    Start-Sleep -Seconds 30
}

Invoke-Pdac @(
    "chromatin-inspect-geo-archive", "--archive", $Archive,
    "--out", "data\studies\training_candidate\GSE99311\archive_inventory.json"
)
Invoke-Pdac @(
    "chromatin-extract-geo-archive", "--archive", $Archive,
    "--output", "data\studies\training_candidate\GSE99311\extracted",
    "--max-unpacked-gb", "25"
)
Invoke-Pdac @("chromatin-geo-metadata", "GSE99311", "--refresh")
Invoke-Pdac @(
    "chromatin-geo-track-specs", "GSE99311",
    "--extracted", "data\studies\training_candidate\GSE99311\extracted"
)
Invoke-Pdac @(
    "chromatin-intervention-pair-plan",
    "--track-index", "data\track_specs\GSE99311\index.json",
    "--out", "data\pair_specs\GSE99311.intervention.json"
)
Invoke-Pdac @(
    "chromatin-compile-index", "--config", "configs\chromatin-local-12gb.json",
    "--track-index", "data\track_specs\GSE99311\index.json",
    "--output", "data\processed\chromatin_gse99311_full_v1",
    "--windows-per-shard", "64", "--negative-keep-probability", "1.0"
)
Invoke-Pdac @(
    "chromatin-materialize-intervention-pairs",
    "--pair-plan", "data\pair_specs\GSE99311.intervention.json",
    "--compiled-root", "data\processed\chromatin_gse99311_full_v1",
    "--output", "data\processed\chromatin_gse99311_paired_v1",
    "--windows-per-shard", "64", "--minimum-overlap-fraction", "0.995"
)
Invoke-Pdac @(
    "chromatin-audit-compiled-splits",
    "--shards", "data/processed/chromatin_gse99311_full_v1/**/*.npz",
    "--out", "results\frozen\gse99311.split-audit.json"
)
Invoke-Pdac @(
    "chromatin-audit-compiled-splits",
    "--shards", "data/processed/chromatin_gse99311_paired_v1/**/*.npz",
    "--out", "results\frozen\gse99311-paired.split-audit.json"
)

$OpenHumanStudies = @(
    "GSE149103", "GSE64557", "GSE272459", "GSE272460",
    "GSE272461", "GSE272462", "GSE272463", "GSE272586"
)
foreach ($Study in $OpenHumanStudies) {
    Invoke-Pdac @(
        "chromatin-study-plan", $Study,
        "--out", "data/manifests/studies/$Study.plan.json"
    )
}

$Plans = @(
    @{ Profile = "configs/chromatin-local-12gb.json"; Out = "results/frozen/chromatin-campaign-local12-plan.json" },
    @{ Profile = "configs/chromatin-scale-24gb.json"; Out = "results/frozen/chromatin-campaign-scale24-plan.json" },
    @{ Profile = "configs/chromatin-scale-48gb.json"; Out = "results/frozen/chromatin-campaign-scale48-plan.json" },
    @{ Profile = "configs/chromatin-scale-80gb.json"; Out = "results/frozen/chromatin-campaign-scale80-plan.json" },
    @{ Profile = "configs/chromatin-ablation-no-progression-graph.json"; Out = "results/frozen/chromatin-ablation-no-progression-graph-plan.json" },
    @{ Profile = "configs/chromatin-ablation-no-domain-invariance.json"; Out = "results/frozen/chromatin-ablation-no-domain-invariance-plan.json" },
    @{ Profile = "configs/chromatin-ablation-direct-long-cnn.json"; Out = "results/frozen/chromatin-ablation-direct-long-plan.json" },
    @{ Profile = "configs/chromatin-ablation-direct-2kb-cnn.json"; Out = "results/frozen/chromatin-ablation-direct-2kb-plan.json" },
    @{ Profile = "configs/chromatin-ablation-mean-only-landmarks.json"; Out = "results/frozen/chromatin-ablation-mean-only-landmarks-plan.json" }
)
foreach ($Plan in $Plans) {
    Invoke-Pdac @(
        "chromatin-plan-campaign", "--campaign", "configs/chromatin-campaign.json",
        "--profile", $Plan.Profile, "--out", $Plan.Out
    )
}

Invoke-Pdac @("predeploy")
