import spotipy
from spotipy.oauth2 import SpotifyOAuth
import numpy as np
from sklearn.preprocessing import StandardScaler


class MusicDiscoverer:
    def __init__(self, client_id, client_secret):
        self.sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
            client_id=client_id,
            client_secret=client_secret,
            redirect_uri="http://127.0.0.1:9090",
            scope="user-top-read user-library-read",
            open_browser=True
        ))

        self.scaler = StandardScaler()
        self.feature_keys = [
            'danceability', 'energy', 'valence', 'tempo',
            'acousticness', 'instrumentalness', 'liveness'
        ]

    def search_songs(self, query, language=None):
        """Search for songs with optional language filter"""
        try:
            # Add language to query if specified
            if language:
                query = f"{query} lang:{language}"

            results = self.sp.search(q=query, type='track', limit=5)
            tracks = results['tracks']['items']

            # If language is specified, try to filter by analyzing track name and artist
            if language and tracks:
                filtered_tracks = []
                for track in tracks:
                    # Get full track details including available markets
                    track_details = self.sp.track(track['id'])
                    # Basic language detection based on markets and track name
                    is_target_language = self._check_language_match(track_details, language)
                    if is_target_language:
                        filtered_tracks.append(track)
                return filtered_tracks[:5]  # Return top 5 matching tracks

            return tracks
        except Exception as e:
            print(f"Search error: {e}")
            return []

    def _check_language_match(self, track, language):
        """Helper method to check if track matches target language"""
        language = language.lower()
        markets = track['available_markets']

        # Dictionary mapping languages to primary markets
        language_markets = {
            'english': ['US', 'GB', 'AU', 'CA'],
            'spanish': ['ES', 'MX', 'AR', 'CO'],
            'french': ['FR', 'CA', 'BE'],
            'german': ['DE', 'AT', 'CH'],
            'japanese': ['JP'],
            'korean': ['KR'],
            'hindi': ['IN']
        }

        if language in language_markets:
            return any(market in markets for market in language_markets[language])
        return True  # If language not in our mapping, don't filter

    def get_features(self, track_id):
        """Get audio features for a track"""
        try:
            features = self.sp.audio_features(track_id)[0]
            return {k: features[k] for k in self.feature_keys}
        except Exception as e:
            print(f"Feature extraction error: {e}")
            return None

    def get_similar_recommendations(self, track_id, n_recommendations=5, target_language=None):
        """Find similar songs with optional language filtering"""
        try:
            # Get seed track features for similarity comparison
            seed_features = self.get_features(track_id)
            if not seed_features:
                return []

            # Get seed track details
            seed_track = self.sp.track(track_id)
            seed_artists = [artist['id'] for artist in seed_track['artists'][:2]]

            # Get recommendations
            recommendations = self.sp.recommendations(
                seed_tracks=[track_id],
                seed_artists=seed_artists[:2],
                target_danceability=seed_features['danceability'],
                target_energy=seed_features['energy'],
                target_valence=seed_features['valence'],
                limit=20
            )

            similar_tracks = []
            for track in recommendations['tracks']:
                # Apply language filtering if specified
                if target_language and not self._check_language_match(track, target_language):
                    continue

                features = self.get_features(track['id'])
                if features:
                    # Calculate similarity score
                    similarity_score = self._calculate_similarity(seed_features, features)

                    similar_tracks.append({
                        'name': track['name'],
                        'artist': track['artists'][0]['name'],
                        'url': track['external_urls']['spotify'],
                        'similarity_score': similarity_score,
                        'key_similarities': self._get_key_similarities(seed_features, features)
                    })

            # Sort by similarity score and get top N
            similar_tracks.sort(key=lambda x: x['similarity_score'], reverse=True)
            return similar_tracks[:n_recommendations]

        except Exception as e:
            print(f"Similar recommendation error: {e}")
            return []

    def get_anti_recommendations(self, track_id, n_recommendations=5):
        """Find opposite songs"""
        try:
            # Get original track features
            orig_features = self.get_features(track_id)
            if not orig_features:
                return []

            # Get recommendations from different genres
            genres = self.sp.recommendation_genre_seeds()['genres'][:5]
            candidates = []

            for genre in genres:
                recs = self.sp.recommendations(seed_genres=[genre], limit=20)
                candidates.extend(recs['tracks'])

            # Get features for all candidates
            candidate_features = []
            valid_candidates = []

            for track in candidates:
                features = self.get_features(track['id'])
                if features:
                    candidate_features.append([features[k] for k in self.feature_keys])
                    valid_candidates.append(track)

            if not valid_candidates:
                return []

            # Scale features and calculate distances
            orig_vector = np.array([orig_features[k] for k in self.feature_keys])
            all_features = np.vstack([orig_vector.reshape(1, -1), candidate_features])
            scaled_features = self.scaler.fit_transform(all_features)

            distances = np.linalg.norm(scaled_features[1:] - scaled_features[0], axis=1)
            most_different_idx = np.argsort(distances)[-n_recommendations:]

            recommendations = []
            for idx in most_different_idx:
                track = valid_candidates[idx]
                features = self.get_features(track['id'])

                differences = []
                for key in self.feature_keys:
                    diff = features[key] - orig_features[key]
                    if abs(diff) > 0.3:
                        differences.append(
                            f"{key} is {'higher' if diff > 0 else 'lower'}"
                        )

                recommendations.append({
                    'name': track['name'],
                    'artist': track['artists'][0]['name'],
                    'url': track['external_urls']['spotify'],
                    'difference_score': distances[idx],
                    'differences': differences
                })

            return recommendations

        except Exception as e:
            print(f"Recommendation error: {e}")
            return []

    def get_quadrant_recommendations(self, track_id):
        """Get recommendations from different quadrants"""
        try:
            # Get original features
            features = self.get_features(track_id)
            if not features:
                return {}

            # Calculate base position
            energy_axis = np.mean([features['energy'], features['danceability']])
            mood_axis = np.mean([features['valence'], features['tempo'] / 200])

            # Define quadrants
            quadrants = {
                'opposite': {'energy': 1 - energy_axis, 'valence': 1 - mood_axis},
                'energy_flip': {'energy': 1 - energy_axis, 'valence': mood_axis},
                'mood_flip': {'energy': energy_axis, 'valence': 1 - mood_axis},
                'similar': {'energy': energy_axis, 'valence': mood_axis}
            }

            results = {}
            for quadrant, targets in quadrants.items():
                recs = self.sp.recommendations(
                    seed_tracks=[track_id],
                    target_energy=targets['energy'],
                    target_valence=targets['valence'],
                    limit=3
                )

                results[quadrant] = []
                for track in recs['tracks']:
                    features = self.get_features(track['id'])
                    if features:
                        results[quadrant].append({
                            'name': track['name'],
                            'artist': track['artists'][0]['name'],
                            'url': track['external_urls']['spotify']
                        })

            return results

        except Exception as e:
            print(f"Quadrant recommendation error: {e}")
            return {}

    def _calculate_similarity(self, features1, features2):
        """Calculate similarity score between two tracks"""
        vector1 = np.array([features1[k] for k in self.feature_keys])
        vector2 = np.array([features2[k] for k in self.feature_keys])

        # Scale features
        scaled = self.scaler.fit_transform(np.vstack([vector1, vector2]))

        # Calculate cosine similarity
        similarity = np.dot(scaled[0], scaled[1]) / (np.linalg.norm(scaled[0]) * np.linalg.norm(scaled[1]))
        return similarity

    def _get_key_similarities(self, features1, features2):
        """Identify key similar features between tracks"""
        similarities = []
        for key in self.feature_keys:
            diff = abs(features1[key] - features2[key])
            if diff < 0.1:  # Features are very similar
                similarities.append(f"Similar {key}")
        return similarities


def main():
    # Your Spotify API credentials
    CLIENT_ID = "id"
    CLIENT_SECRET = "secret"

    discoverer = MusicDiscoverer(CLIENT_ID, CLIENT_SECRET)

    while True:
        print("\nMusic Discovery Tool")
        print("1. Similar Recommendations")
        print("2. Anti-Recommendations (maximally different songs)")
        print("3. Quadrant Discovery")
        print("4. Exit")

        choice = input("\nSelect mode (1-4): ")

        if choice == '4':
            break

        try:
            # Get language preference
            print("\nAvailable languages: english, spanish, french, german, japanese, korean, hindi")
            print("Press Enter to skip language filter")
            language = input("Enter preferred language (or press Enter): ").lower()
            if language == "":
                language = None

            query = input("\nEnter song name to search: ")
            results = discoverer.search_songs(query, language)

            if not results:
                print("No songs found!")
                continue

            print("\nFound these songs:")
            for i, track in enumerate(results, 1):
                print(f"{i}. {track['name']} by {track['artists'][0]['name']}")

            track_choice = int(input("\nSelect a song (1-5): ")) - 1
            if track_choice < 0 or track_choice >= len(results):
                print("Invalid selection!")
                continue

            selected_track = results[track_choice]

            if choice == '1':
                print("\nFinding similar songs...")
                recommendations = discoverer.get_similar_recommendations(
                    selected_track['id'],
                    target_language=language
                )

                if recommendations:
                    print("\nSimilar Songs:")
                    for i, rec in enumerate(recommendations, 1):
                        print(f"\n{i}. {rec['name']} by {rec['artist']}")
                        print(f"Similarity Score: {rec['similarity_score']:.2f}")
                        if rec['key_similarities']:
                            print("Key Similarities:")
                            for sim in rec['key_similarities']:
                                print(f"- {sim}")
                        print(f"Listen: {rec['url']}")
                else:
                    print("Couldn't find recommendations!")

            elif choice == '2':
                print("\nFinding maximally different songs...")
                recommendations = discoverer.get_anti_recommendations(selected_track['id'])

                if recommendations:
                    print("\nRecommendations:")
                    for i, rec in enumerate(recommendations, 1):
                        print(f"\n{i}. {rec['name']} by {rec['artist']}")
                        print(f"Difference Score: {rec['difference_score']:.2f}")
                        print("Key Differences:")
                        for diff in rec['differences']:
                            print(f"- {diff}")
                        print(f"Listen: {rec['url']}")
                else:
                    print("Couldn't find recommendations!")

            elif choice == '3':
                print("\nFinding songs in different quadrants...")
                recommendations = discoverer.get_quadrant_recommendations(selected_track['id'])

                if recommendations:
                    print("\nRecommendations by Quadrant:")
                    for quadrant, tracks in recommendations.items():
                        if tracks:
                            print(f"\n[{quadrant.upper()}]")
                            for i, track in enumerate(tracks, 1):
                                print(f"{i}. {track['name']} by {track['artist']}")
                                print(f"Listen: {track['url']}")
                else:
                    print("Couldn't find recommendations!")

        except Exception as e:
            print(f"\nError: {e}")
            continue


if __name__ == "__main__":
    main()