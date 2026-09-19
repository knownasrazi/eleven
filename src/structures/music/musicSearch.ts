/**
 * Credits: The OpenUwU Project
 * Owners: @priyanshu @prayag
 * Author:  @bre4d777 and @mooncarli
 * github.com/openUwU/
 */

import { logger } from "../../utils/logger.js";
import type { Manager } from "./Manager.js";
import type { PartialQueueTrack, SearchResultNormalized } from "./types.js";
import { SearchSource } from "./types.js";

interface ScoreCache {
	tToks?: string[];
	pToks?: string;
	phoneticSet?: Set<string>;
	splitArtistTitle?: { artist: string; song: string };
	artistTokens?: string[];
	songTokens?: string[];
	authorTokens?: string[];
	artistTokenSet?: Set<string>;
	songTokenSet?: Set<string>;
	authorTokenSet?: Set<string>;
}

const scoreCaches = new WeakMap<PartialQueueTrack, ScoreCache>();

function cacheFor(track: PartialQueueTrack): ScoreCache {
	let scoreCache = scoreCaches.get(track);
	if (!scoreCache) {
		scoreCache = {};
		scoreCaches.set(track, scoreCache);
	}
	return scoreCache;
}

export interface ResolvedMatch {
	track: PartialQueueTrack;
	score: number;
	confidence: "perfect" | "high" | "medium" | "low";
	source: "spotify" | "mirror" | "ytm" | "yt";
}

export interface ResolveQueryResult {
	type: "track" | "playlist" | "empty" | "error";
	tracks: PartialQueueTrack[];
	playlistName?: string | null;
}

interface CanonicalTrack {
	title: string;
	artist: string | null | undefined;
	duration: number | null | undefined;
}

const IS_LINK = /^https?:\/\//i;

const IS_NON_ENGLISH = /[^ -~]/;

const BAD_WORDS = /\b(remix|nightcore|bass boosted|8d|slowed|reverb|cover|karaoke)\b/i;
const FEAT_REGEX = /\b(feat|ft|featuring)\b/i;
const CLEAN_WORDS = /\b(official|video|lyrics|audio|music)\b/i;
const OFFICIAL_HINT = /\b(official|vevo|topic|records|music)\b/i;

const MIRROR_SOURCES = [SearchSource.JioSaavn, SearchSource.Deezer] as const;
const MIRROR_THRESHOLD = 0.72;
const MIRROR_CANDIDATES = 4;

const SP_CANDIDATES = 6;
const YTM_CANDIDATES = 5;
const YT_CANDIDATES = 5;
const MIN_YT_SCORE = 0.18;

const SP_TIMEOUT_MS = 10_000;
const MIRROR_TIMEOUT_MS = 7_000;

function withTimeout<T>(promise: Promise<T>, ms: number): Promise<T> {
	let timer: ReturnType<typeof setTimeout> | undefined;
	const timeout = new Promise<T>((_, reject) => {
		timer = setTimeout(() => reject(new Error(`Timed out after ${ms}ms`)), ms);
	});
	return Promise.race([promise, timeout]).finally(() => {
		if (timer !== undefined) clearTimeout(timer);
	});
}

function clamp(n: number): number {
	return n > 1 ? 1 : n < 0 ? 0 : n;
}

function reportError(tag: string, error: unknown): void {
	logger.error(tag, error instanceof Error ? error.message : String(error), error as Error);
}

function tracksOf(result: SearchResultNormalized): PartialQueueTrack[] {
	return result.type === "track" || result.type === "search" || result.type === "playlist"
		? result.tracks
		: [];
}

function normalize(s: string | null | undefined): string {
	if (!s) return "";
	return s
		.toLowerCase()
		.replace(FEAT_REGEX, " ")
		.replace(CLEAN_WORDS, "")
		.replace(/[^\p{L}\p{N}\s]/gu, " ")
		.replace(/\s+/g, " ")
		.trim();
}

export function tokenize(s: string | null | undefined): string[] {
	if (!s) return [];
	return normalize(s).split(" ").filter(Boolean);
}

function phonetic(s: string | null | undefined): string {
	return (
		s
			?.toLowerCase()
			.replace(/aa/g, "a")
			.replace(/ee/g, "i")
			.replace(/oo/g, "u")
			.replace(/ph/g, "f")
			.replace(/ck/g, "k")
			.replace(/sh/g, "s")
			.replace(/iya$/g, "ia")
			.replace(/iye$/g, "ie")
			.replace(/[^a-z0-9\s]/g, "")
			.replace(/\s+/g, " ")
			.trim() || ""
	);
}

function tokenOverlap(aToks: string[], bToks: string[]): number {
	if (!aToks.length || !bToks.length) return 0;
	const bSet = new Set(bToks);
	let matches = 0;
	for (const w of aToks) if (bSet.has(w)) matches++;
	return matches / Math.max(aToks.length, bToks.length);
}

function wordBagFromTokens(aToks: string[], bToks: string[]): number {
	if (!aToks.length) return 0;
	const bSet = new Set(bToks);
	let matchCount = 0;
	for (const token of aToks) if (bSet.has(token)) matchCount++;
	return matchCount / aToks.length;
}

function durationSim(a: number | null | undefined, b: number | null | undefined): number {
	if (!a || !b) return 0;
	const diff = Math.abs(a - b);
	if (diff < 2000) return 1.0;
	if (diff < 5000) return 0.7;
	if (diff < 10_000) return 0.4;
	return 0;
}

function durationSanity(ms: number | null | undefined): number {
	if (!ms) return 0.8;
	if (ms < 40_000) return 0.2;
	if (ms > 10 * 60_000) return 0.5;
	return 1.0;
}

function cleanArtist(a: string | null | undefined): string {
	return a
		? a
				.replace(/\s*-\s*topic$/i, "")
				.replace(/vevo$/i, "")
				.replace(/\s*official\s*$/i, "")
				.trim()
		: "";
}

function splitArtistTitle(title: string | null | undefined): { artist: string; song: string } {
	if (!title) return { artist: "", song: "" };
	const separators = [" - ", " – ", " — ", " | ", " : "];
	for (const separator of separators) {
		const index = title.indexOf(separator);
		if (index !== -1) {
			return {
				artist: title.slice(0, index).trim(),
				song: title.slice(index + separator.length).trim(),
			};
		}
	}
	return { artist: "", song: title };
}

function levenshtein(a: string, b: string, cap = Number.POSITIVE_INFINITY): number {
	if (Math.abs(a.length - b.length) > cap) return cap + 1;
	let prev = Array.from({ length: a.length + 1 }, (_, i) => i);
	let curr = new Array<number>(a.length + 1);
	for (let j = 1; j <= b.length; j++) {
		curr[0] = j;
		let rowMin = j;
		for (let i = 1; i <= a.length; i++) {
			const cost = (prev[i - 1] ?? 0) + (a[i - 1] === b[j - 1] ? 0 : 1);
			const val = Math.min((curr[i - 1] ?? 0) + 1, (prev[i] ?? 0) + 1, cost);
			curr[i] = val;
			if (val < rowMin) rowMin = val;
		}
		if (rowMin > cap) return cap + 1;
		[prev, curr] = [curr, prev];
	}
	return prev[a.length] ?? 0;
}

function artistTitleBonus(queryTokens: string[], title: string, track: PartialQueueTrack): number {
	const cache = cacheFor(track);
	cache.splitArtistTitle ??= splitArtistTitle(title);
	const split = cache.splitArtistTitle;
	cache.artistTokens ??= tokenize(cleanArtist(split.artist));
	const artistTokens = cache.artistTokens;
	cache.songTokens ??= tokenize(split.song);
	const songTokens = cache.songTokens;
	cache.authorTokens ??= tokenize(cleanArtist(track.info.author ?? ""));
	const authorTokens = cache.authorTokens;

	cache.artistTokenSet ??= new Set(artistTokens);
	const artistSet = cache.artistTokenSet;
	cache.songTokenSet ??= new Set(songTokens);
	const songSet = cache.songTokenSet;
	cache.authorTokenSet ??= new Set(authorTokens);
	const authorSet = cache.authorTokenSet;

	let artistMatches = 0;
	let songMatches = 0;
	let authorMatches = 0;

	for (const token of queryTokens) {
		if (artistSet.has(token)) artistMatches++;
		if (songSet.has(token)) songMatches++;
		if (authorSet.has(token)) authorMatches++;
	}

	const maxTokens = Math.max(queryTokens.length, 1);
	return (
		(artistMatches / maxTokens) * 0.1 +
		(songMatches / maxTokens) * 0.1 +
		(authorMatches / maxTokens) * 0.2
	);
}

function substringBonus(qToks: string[], tToks: string[]): number {
	if (!qToks.length) return 0;
	const qStr = qToks.join(" ");
	const tStr = tToks.join(" ");
	if (tStr.includes(qStr)) return 0.15;
	const firstQTok = qToks[0];
	if (firstQTok !== undefined && tToks.some((w) => w.includes(firstQTok))) return 0.06;
	return 0;
}

function orderBonus(qToks: string[], tToks: string[]): number {
	let matchCount = 0;
	for (let i = 0; i < Math.min(qToks.length, tToks.length); i++) {
		if (qToks[i] === tToks[i]) matchCount++;
	}
	return matchCount / Math.max(qToks.length, 1);
}

function officialBoost(track: PartialQueueTrack): number {
	const author = track.info.author ?? "";
	const title = track.info.title ?? "";
	return OFFICIAL_HINT.test(author) || OFFICIAL_HINT.test(title) ? 0.04 : 0;
}

function shortQueryScore(query: string, track: PartialQueueTrack): number {
	const lowerCaseQuery = query.trim().toLowerCase();
	const title = (track.info.title || "").toLowerCase();
	const author = cleanArtist(track.info.author).toLowerCase();

	const normQ = normalize(lowerCaseQuery);
	const normT = normalize(title);
	if (normT === normQ) return 1.0;
	if (normT.startsWith(`${normQ} `) || normT === normQ) return 0.95;

	const rawQ = lowerCaseQuery.replace(/\s+/g, " ").trim();
	if (title.startsWith(rawQ)) return 0.92;

	const qParts = rawQ.split(/\s+/);
	if (qParts.length >= 2) {
		const inTitle = qParts.filter((part) => normT.includes(normalize(part)));
		const inAuthor = qParts.filter((part) => normalize(author).includes(normalize(part)));
		const covered = new Set([...inTitle, ...inAuthor]);
		if (covered.size === qParts.length && inTitle.length > 0) return 0.9;
	}

	const lev = levenshtein(normQ, normT, 3);
	if (lev === 0) return 0.88;
	if (lev === 1) return 0.72;
	if (lev === 2) return 0.55;
	return 0;
}

export function scoreTrack(
	query: string,
	track: PartialQueueTrack,
	avgDurationMs: number | null = null,
): number {
	if (!query || !track?.info?.title) return 0;

	const title = track.info.title || "";
	const duration = track.info.length;
	const titleLower = title.toLowerCase();
	const qTrimmed = query.trim();
	const isShort = tokenize(query).length <= 1;

	if (isShort) {
		const base = shortQueryScore(qTrimmed, track);
		const durS = durationSanity(duration) * 0.05;
		const pen = BAD_WORDS.test(titleLower) ? 0.2 : 0;
		return clamp(base + durS - pen);
	}

	const cache = cacheFor(track);
	const qToks = tokenize(qTrimmed);
	cache.tToks ??= tokenize(title);
	const tToks = cache.tToks;

	const tokenScore = tokenOverlap(qToks, tToks);
	const bagScore = wordBagFromTokens(qToks, tToks);

	let fuzzyScore = 0;
	if (tokenScore > 0.45) {
		const qStr = qToks.join(" ");
		const tStr = tToks.join(" ");
		if (qStr.length <= 50 && tStr.length <= 70) {
			const dist = levenshtein(qStr, tStr, Math.ceil(qStr.length * 0.5));
			fuzzyScore = dist <= qStr.length * 0.5 ? 1 - dist / Math.max(qStr.length, tStr.length) : 0;
		}
	}

	let phoneticScore = 0;
	if (qToks.length >= 2 && tokenScore >= 0.25) {
		const pQ = phonetic(qTrimmed);
		cache.pToks ??= phonetic(title);
		const pT = cache.pToks;
		const pQToks = pQ.split(" ").filter(Boolean);
		cache.phoneticSet ??= new Set(pT.split(" ").filter(Boolean));
		const pTSet = cache.phoneticSet;
		const pHits = pQToks.filter((w) => pTSet.has(w)).length;
		phoneticScore = pQToks.length ? (pHits / pQToks.length) * 0.1 : 0;
	}

	const score =
		tokenScore * 0.38 +
		bagScore * 0.2 +
		fuzzyScore * 0.16 +
		phoneticScore +
		substringBonus(qToks, tToks) +
		orderBonus(qToks, tToks) * 0.05 +
		artistTitleBonus(qToks, title, track) +
		officialBoost(track) +
		(avgDurationMs ? durationSim(duration, avgDurationMs) * 0.08 : 0) +
		durationSanity(duration) * 0.06 -
		(BAD_WORDS.test(titleLower) ? 0.25 : 0);

	return clamp(score);
}

export function scoreMirrorMatch(canonical: CanonicalTrack, track: PartialQueueTrack): number {
	if (!canonical || !track?.info) return 0;

	const titleSim = (() => {
		const normalizedCanonicalTitle = normalize(canonical.title);
		const normalizedTrackTitle = normalize(track.info.title || "");
		if (normalizedCanonicalTitle === normalizedTrackTitle) return 1.0;
		if (
			normalizedCanonicalTitle.includes(normalizedTrackTitle) ||
			normalizedTrackTitle.includes(normalizedCanonicalTitle)
		)
			return 0.85;
		const aToks = tokenize(canonical.title);
		const bToks = tokenize(track.info.title || "");
		const overlap = tokenOverlap(aToks, bToks);
		const dist = levenshtein(
			normalizedCanonicalTitle,
			normalizedTrackTitle,
			Math.ceil(normalizedCanonicalTitle.length * 0.4),
		);
		const lev =
			dist <= normalizedCanonicalTitle.length * 0.4
				? 1 - dist / Math.max(normalizedCanonicalTitle.length, normalizedTrackTitle.length)
				: 0;
		return overlap * 0.7 + lev * 0.3;
	})();

	const artistSim = (() => {
		const normalizedCanonicalArtist = normalize(cleanArtist(canonical.artist));
		const normalizedTrackAuthor = normalize(cleanArtist(track.info.author));
		if (!normalizedCanonicalArtist || !normalizedTrackAuthor) return 0.5;
		if (normalizedCanonicalArtist === normalizedTrackAuthor) return 1.0;
		if (
			normalizedCanonicalArtist.includes(normalizedTrackAuthor) ||
			normalizedTrackAuthor.includes(normalizedCanonicalArtist)
		)
			return 0.8;
		return tokenOverlap(
			normalizedCanonicalArtist.split(" ").filter(Boolean),
			normalizedTrackAuthor.split(" ").filter(Boolean),
		);
	})();

	const durSim = durationSim(canonical.duration, track.info.length);

	return clamp(titleSim * 0.52 + artistSim * 0.33 + durSim * 0.15);
}

export function scoreWithPosition(
	query: string,
	track: PartialQueueTrack,
	index: number,
	total: number,
	avgDurationMs: number | null,
): number {
	const rel = scoreTrack(query, track, avgDurationMs);
	const qLen = query.trim().length;

	const posWeight = qLen <= 3 ? 0.05 : qLen <= 6 ? 0.12 : 0.18;
	const pos = 1 - index / Math.max(total, 1);

	return clamp(rel * (1 - posWeight) + pos * posWeight);
}

export function avgDuration(tracks: PartialQueueTrack[]): number | null {
	const durations = tracks.map((t) => t.info?.length).filter((d): d is number => Boolean(d));
	if (!durations.length) return null;
	return durations.reduce((a, b) => a + b, 0) / durations.length;
}

export function confidenceLevel(score: number): ResolvedMatch["confidence"] {
	if (score >= 0.9) return "perfect";
	if (score >= 0.78) return "high";
	if (score >= 0.62) return "medium";
	return "low";
}

async function resolveSpotifyOracle(
	manager: Manager,
	query: string,
): Promise<{ track: PartialQueueTrack; score: number; canonical: CanonicalTrack } | null> {
	logger.debug("MusicSearch:SpotifyOracle", `Searching Spotify for "${query}"`);

	let result: SearchResultNormalized;
	try {
		result = await withTimeout(manager.search(query, SearchSource.Spotify), SP_TIMEOUT_MS);
	} catch (error) {
		reportError("SpotifyOracle", error);
		return null;
	}

	const tracks = tracksOf(result);
	logger.debug("MusicSearch:SpotifyOracle", `Got ${tracks.length} tracks (type=${result.type})`);
	if (!tracks.length) return null;

	const candidates = tracks.slice(0, SP_CANDIDATES);
	const averageDuration = avgDuration(candidates);

	let bestScore = -1;
	let bestTrack: PartialQueueTrack | null = null;

	for (let i = 0; i < candidates.length; i++) {
		const candidate = candidates[i];
		if (!candidate?.info) continue;
		const score = scoreWithPosition(query, candidate, i, candidates.length, averageDuration);
		logger.debug(
			"MusicSearch:SpotifyOracle",
			`  [${i}] score=${score.toFixed(3)} title="${candidate.info.title}" author="${candidate.info.author}"`,
		);
		if (score > bestScore) {
			bestScore = score;
			bestTrack = candidate;
		}
		if (bestScore >= 0.85) {
			logger.debug("MusicSearch:SpotifyOracle", `  Early exit at index ${i} (score >= 0.85)`);
			break;
		}
	}

	if (!bestTrack || bestScore < 0.2) {
		logger.debug(
			"MusicSearch:SpotifyOracle",
			`No usable oracle (bestScore=${bestScore.toFixed(3)})`,
		);
		return null;
	}

	logger.debug(
		"MusicSearch:SpotifyOracle",
		`Oracle resolved: "${bestTrack.info.title}" score=${bestScore.toFixed(3)}`,
	);

	return {
		track: bestTrack,
		score: bestScore,
		canonical: {
			title: bestTrack.info.title,
			artist: bestTrack.info.author,
			duration: bestTrack.info.length,
		},
	};
}

async function resolveMirrorTrack(
	manager: Manager,
	canonical: CanonicalTrack,
): Promise<{ source: SearchSource; track: PartialQueueTrack; score: number } | null> {
	if (!canonical) return null;

	const mirrorQuery = canonical.artist ? `${canonical.title} ${canonical.artist}` : canonical.title;
	logger.debug(
		"MusicSearch:Mirror",
		`Searching mirror sources for canonical "${canonical.title}" by "${canonical.artist ?? "?"}" — mirrorQuery="${mirrorQuery}"`,
	);

	const results = await Promise.all(
		MIRROR_SOURCES.map(async (source) => {
			try {
				const result = await withTimeout(manager.search(mirrorQuery, source), MIRROR_TIMEOUT_MS);
				const tracks = tracksOf(result).slice(0, MIRROR_CANDIDATES);
				logger.debug(`MusicSearch:Mirror:${source}`, `Got ${tracks.length} candidates`);
				if (!tracks.length) return null;

				let bestScore = -1;
				let bestTrack: PartialQueueTrack | null = null;

				for (const t of tracks) {
					const score = scoreMirrorMatch(canonical, t);
					logger.debug(
						`MusicSearch:Mirror:${source}`,
						`  score=${score.toFixed(3)} title="${t.info.title}" author="${t.info.author}"`,
					);
					if (score > bestScore) {
						bestScore = score;
						bestTrack = t;
					}
					if (bestScore >= 0.85) {
						logger.debug(`MusicSearch:Mirror:${source}`, "  Early exit (score >= 0.85)");
						break;
					}
				}

				if (!bestTrack || bestScore < MIRROR_THRESHOLD) {
					logger.debug(
						`MusicSearch:Mirror:${source}`,
						`Below threshold (bestScore=${bestScore.toFixed(3)} threshold=${MIRROR_THRESHOLD})`,
					);
					return null;
				}
				logger.debug(
					`MusicSearch:Mirror:${source}`,
					`Matched "${bestTrack.info.title}" score=${bestScore.toFixed(3)}`,
				);
				return { source, track: bestTrack, score: bestScore };
			} catch (error) {
				reportError(`Mirror:${source}`, error);
				return null;
			}
		}),
	);

	for (const result of results) if (result) return result;
	logger.debug("MusicSearch:Mirror", "No mirror source met the threshold");
	return null;
}

function sanitizeYoutubeQuery(query: string): { cleaned: string; isDirty: boolean } {
	const cleaned = query
		.replace(/[([].+?[)\]]/g, "")
		.replace(/\s*-\s*official.*/gi, "")
		.replace(/\s*-\s*lyric.*/gi, "")
		.replace(/[•·].*/g, "")
		.replace(/\d+\s*(months?|years?|days?|hours?|weeks?)\s*ago/gi, "")
		.replace(/\s+/g, " ")
		.trim();

	const isDirty =
		/[•·]/.test(query) ||
		/\d+[KMB]?\s*views/i.test(query) ||
		/(months?|years?|days?)\s*ago/i.test(query);

	return { cleaned, isDirty };
}

async function searchYoutubeSource(
	manager: Manager,
	source: SearchSource,
	query: string,
): Promise<PartialQueueTrack[]> {
	try {
		const result = await manager.search(query, source);
		return tracksOf(result);
	} catch (error) {
		reportError(`YoutubeFallback:${source}`, error);
		return [];
	}
}

function pickBestCandidate(
	query: string,
	tracks: PartialQueueTrack[],
	maxCandidates: number,
): { track: PartialQueueTrack | null; score: number } {
	const candidates = tracks.slice(0, maxCandidates);
	const avg = avgDuration(candidates);
	let bestScore = -1;
	let bestTrack: PartialQueueTrack | null = null;

	for (let i = 0; i < candidates.length; i++) {
		const candidateTrack = candidates[i];
		if (!candidateTrack?.info) continue;
		const score = scoreWithPosition(query, candidateTrack, i, candidates.length, avg);
		if (score > bestScore) {
			bestScore = score;
			bestTrack = candidateTrack;
		}
		if (bestScore >= 0.93) break;
	}

	return { track: bestTrack, score: bestScore };
}

async function resolveYoutubeFallback(
	manager: Manager,
	query: string,
): Promise<{ track: PartialQueueTrack; score: number; source: "ytm" | "yt" } | null> {
	const { cleaned, isDirty } = sanitizeYoutubeQuery(query);
	const queries = isDirty && cleaned !== query ? [query, cleaned] : [query];

	const ytmResults = await Promise.all(
		queries.map((q) => searchYoutubeSource(manager, SearchSource.YouTubeMusic, q)),
	);
	const ytmPool = ytmResults.flat();

	if (ytmPool.length) {
		const { track, score } = pickBestCandidate(query, ytmPool, YTM_CANDIDATES);
		if (track && score >= MIN_YT_SCORE) return { track, score, source: "ytm" };
	}

	const ytResults = await Promise.all(
		queries.map((q) => searchYoutubeSource(manager, SearchSource.YouTube, q)),
	);
	const ytPool = ytResults.flat();

	if (ytPool.length) {
		const { track, score } = pickBestCandidate(query, ytPool, YT_CANDIDATES);
		if (track && score >= MIN_YT_SCORE) return { track, score, source: "yt" };
		if (ytPool[0]?.info) return { track: ytPool[0], score: 0, source: "yt" };
	}

	return null;
}

async function resolveTextQuery(
	manager: Manager,
	query: string,
	onFallback?: () => void | Promise<void>,
): Promise<ResolvedMatch | null> {
	const oracle = await resolveSpotifyOracle(manager, query);

	if (oracle) {
		const mirror = await resolveMirrorTrack(manager, oracle.canonical);
		if (mirror) {
			return {
				track: mirror.track,
				score: mirror.score,
				confidence: confidenceLevel(mirror.score),
				source: "mirror",
			};
		}
		return {
			track: oracle.track,
			score: oracle.score,
			confidence: confidenceLevel(oracle.score),
			source: "spotify",
		};
	}

	if (onFallback) await onFallback();

	const yt = await resolveYoutubeFallback(manager, query);
	if (yt) {
		return {
			track: yt.track,
			score: yt.score,
			confidence: confidenceLevel(yt.score),
			source: yt.source,
		};
	}

	return null;
}

async function resolveNonEnglishTextQuery(
	manager: Manager,
	query: string,
	onFallback?: () => void | Promise<void>,
): Promise<ResolvedMatch | null> {
	try {
		const jsResult = await manager.search(query, SearchSource.JioSaavn);
		const jsTracks = tracksOf(jsResult);
		const jsFirst = jsTracks[0];
		if (jsFirst !== undefined) {
			return { track: jsFirst, score: 1, confidence: "high", source: "mirror" };
		}
	} catch {
		// empty
	}

	return resolveTextQuery(manager, query, onFallback);
}

export interface ResolveQueryOptions {
	manager: Manager;
	query: string;
	onFallback?: () => void | Promise<void>;
}

export async function resolveQuery({
	manager,
	query,
	onFallback,
}: ResolveQueryOptions): Promise<ResolveQueryResult> {
	const trimmed = query?.trim();
	if (!trimmed) {
		logger.debug("MusicSearch:resolveQuery", "Empty query — returning empty");
		return { type: "empty", tracks: [] };
	}

	if (IS_LINK.test(trimmed)) {
		logger.debug("MusicSearch:resolveQuery", `Detected URL — resolving directly: ${trimmed}`);
		try {
			const result = await manager.search(trimmed, SearchSource.Direct);
			logger.debug(
				"MusicSearch:resolveQuery",
				`Direct URL result: type=${result.type}${result.type === "playlist" ? ` name="${(result as { playlistName?: string | null }).playlistName ?? "?"}" tracks=${tracksOf(result).length}` : ""}`,
			);
			if (result.type === "empty") return { type: "empty", tracks: [] };
			if (result.type === "error") return { type: "error", tracks: [] };
			if (result.type === "playlist") {
				return { type: "playlist", tracks: result.tracks, playlistName: result.playlistName };
			}
			return { type: "track", tracks: result.tracks };
		} catch (err) {
			logger.debug(
				"MusicSearch:resolveQuery",
				`Direct URL search threw: ${(err as Error).message}`,
			);
			return { type: "error", tracks: [] };
		}
	}

	const isNonEnglish = IS_NON_ENGLISH.test(trimmed);
	logger.debug(
		"MusicSearch:resolveQuery",
		`Text query — nonEnglish=${isNonEnglish} query="${trimmed}"`,
	);

	const match = isNonEnglish
		? await resolveNonEnglishTextQuery(manager, trimmed, onFallback)
		: await resolveTextQuery(manager, trimmed, onFallback);

	if (!match) {
		logger.debug("MusicSearch:resolveQuery", `No match found for query "${trimmed}"`);
		return { type: "empty", tracks: [] };
	}

	logger.debug(
		"MusicSearch:resolveQuery",
		`Resolved "${trimmed}" → "${match.track.info.title}" by "${match.track.info.author}" source=${match.source} score=${match.score.toFixed(3)} confidence=${match.confidence}`,
	);
	return { type: "track", tracks: [match.track] };
}

export default resolveQuery;
