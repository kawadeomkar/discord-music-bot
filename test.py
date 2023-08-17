import asyncio
import spotify
import sources


async def main():
    print('Hello ...')
    surl = "https://open.spotify.com/track/23gcQr3NRKzLXsP9H5jFQ1?si=8d255931ca894543"
    surl2 = "https://open.spotify.com/playlist/2jEvNUvrWJZaYGlxVGdbp3?si=3c5a640ca10e4271"
    surl3 = "https://open.spotify.com/playlist/0ObdckZXsQInIGzdWdcaqu?si=1fa714b3d6f44f15"
    yurl = "https://youtu.be/PNpf1QO7-Hg?t=9&ee=lol"
    yurl2 = "https://youtu.be/YOx3XlQekpA"

    ytsearch_source = sources.parse_url("hey", "-p hey there fella")
    sp_source = sources.parse_url(surl3, f"-p {surl3}")
    yt_source = sources.parse_url(yurl, f"-p {yurl}")
    yt2_source = sources.parse_url(yurl2, f"-p {yurl2}")

    spot = spotify.Spotify()
    print(sp_source)
    print(spot.auth_token)
    titles = await spot.playlist(sp_source.id)
    print(titles)

    #print(yt_source)

# Python 3.7+
asyncio.run(main())